import operator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from runspace.src.quantization.activation_stage_planner import (
    NodeRole,
    StageKind,
    UnsupportedNodeError,
    plan_activation_stages,
)


class ConvReluConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, inputs):
        return self.conv2(self.relu(self.conv1(inputs)))


class ResidualFanout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, inputs):
        pre_activation = self.conv1(inputs)
        branch = self.conv2(self.relu(pre_activation))
        return branch + pre_activation


class AttentionPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        probabilities = F.softmax(scores, dim=-1)
        probabilities = self.dropout(probabilities)
        return torch.matmul(probabilities, value)


class UnknownTensorOperation(nn.Module):
    def forward(self, inputs):
        return torch.sin(inputs)


class SigmoidPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        return self.sigmoid(self.linear(inputs))


class UnbindPath(nn.Module):
    def forward(self, inputs):
        left, right = inputs.unbind(0)
        return left + right


class StochasticDepthPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = StochasticDepth(p=0.2, mode="row")

    def forward(self, inputs):
        return self.drop(inputs)


def test_conv_relu_conv_fuses_boundary_after_relu():
    plan = plan_activation_stages(ConvReluConv().eval())

    conv1_stage = plan.stage_for_node("conv1")
    relu_stage = plan.stage_for_node("relu")
    conv2_stage = plan.stage_for_node("conv2")

    assert relu_stage.stage_id == conv1_stage.stage_id
    assert conv1_stage.kind == StageKind.COMPUTE
    assert conv1_stage.node_names == ("conv1", "relu")
    assert conv1_stage.output_node == "relu"
    assert conv1_stage.is_unsigned
    assert conv1_stage.unsigned_source == "relu"
    assert conv1_stage.consumer_nodes == ("conv2",)
    assert conv2_stage.input_stage_ids == (conv1_stage.stage_id,)
    assert plan.node_roles["relu"] == NodeRole.ACTIVATION


def test_residual_fanout_keeps_pre_activation_boundary():
    plan = plan_activation_stages(ResidualFanout().eval())

    conv1_stage = plan.stage_for_node("conv1")
    relu_stage = plan.stage_for_node("relu")
    add_node = next(
        node
        for node in plan.graph_module.graph.nodes
        if node.op == "call_function" and node.target is operator.add
    )
    add_stage = plan.stage_for_node(add_node.name)

    assert conv1_stage.output_node == "conv1"
    assert conv1_stage.node_names == ("conv1",)
    assert conv1_stage.has_fanout
    assert set(conv1_stage.consumer_nodes) == {"relu", add_node.name}
    assert relu_stage.stage_id != conv1_stage.stage_id
    assert relu_stage.kind == StageKind.ACTIVATION
    assert relu_stage.is_unsigned
    assert conv1_stage.stage_id in add_stage.input_stage_ids


def test_softmax_boundary_survives_inference_dropout_passthrough():
    plan = plan_activation_stages(AttentionPath().eval())
    matmul_nodes = [
        node
        for node in plan.graph_module.graph.nodes
        if node.op == "call_function" and node.target is torch.matmul
    ]
    assert len(matmul_nodes) == 2
    score_matmul, value_matmul = matmul_nodes
    softmax_node = next(
        node
        for node in plan.graph_module.graph.nodes
        if node.op == "call_function" and node.target is F.softmax
    )

    score_stage = plan.stage_for_node(score_matmul.name)
    softmax_stage = plan.stage_for_node(softmax_node.name)
    value_stage = plan.stage_for_node(value_matmul.name)

    assert softmax_stage.stage_id == score_stage.stage_id
    assert score_stage.output_node == softmax_node.name
    assert score_stage.is_unsigned
    assert score_stage.unsigned_source == "softmax"
    assert score_stage.passthrough_nodes == ("dropout",)
    assert score_stage.consumer_nodes == (value_matmul.name,)
    assert score_stage.stage_id in value_stage.input_stage_ids
    assert plan.node_roles["dropout"] == NodeRole.TRANSPARENT


def test_training_dropout_is_a_new_producer_stage():
    plan = plan_activation_stages(AttentionPath().train())

    dropout_stage = plan.stage_for_node("dropout")
    assert dropout_stage.kind == StageKind.COMPUTE
    assert plan.node_roles["dropout"] == NodeRole.COMPUTE
    assert dropout_stage.output_node == "dropout"


def test_unknown_tensor_operation_fails_closed():
    with pytest.raises(UnsupportedNodeError, match="torch.sin|sin"):
        plan_activation_stages(UnknownTensorOperation().eval())


def test_sigmoid_is_an_unsigned_activation_boundary():
    plan = plan_activation_stages(SigmoidPath().eval())

    linear_stage = plan.stage_for_node("linear")
    sigmoid_stage = plan.stage_for_node("sigmoid")
    assert sigmoid_stage.stage_id == linear_stage.stage_id
    assert linear_stage.output_node == "sigmoid"
    assert linear_stage.is_unsigned
    assert linear_stage.unsigned_source == "sigmoid"


def test_unbind_is_a_transparent_tensor_partition():
    plan = plan_activation_stages(UnbindPath().eval())

    assert plan.node_roles["unbind"] == NodeRole.TRANSPARENT
    assert plan.stage_for_node("inputs").consumer_nodes == ("add",)


def test_stochastic_depth_matches_dropout_transport_semantics():
    eval_plan = plan_activation_stages(StochasticDepthPath().eval())
    train_plan = plan_activation_stages(StochasticDepthPath().train())

    assert eval_plan.node_roles["stochastic_depth"] == NodeRole.TRANSPARENT
    assert train_plan.node_roles["stochastic_depth"] == NodeRole.COMPUTE
    assert train_plan.stage_for_node("stochastic_depth").kind == StageKind.COMPUTE


def test_opaque_multihead_attention_is_rejected():
    class OpaqueAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(8, 2, batch_first=True)

        def forward(self, value):
            return self.attention(value, value, value, need_weights=False)[0]

    with pytest.raises(UnsupportedNodeError, match="must be lowered"):
        plan_activation_stages(OpaqueAttention().eval())
