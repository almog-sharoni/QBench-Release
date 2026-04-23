"""Extract only the frames referenced by scannet_test_pairs_with_gt.txt from
raw .sens files, writing JPEG bytes directly (no decompress/re-encode)."""
import os, struct, json, sys

NEEDED = json.load(open('/tmp/scannet_extract/needed.json'))
ROOT_IN  = '/data/scannet/scans_test'
ROOT_OUT = '/data/scannet/scans_test'   # in-place: write {scene}/sens/frame-NNNNNN.color.jpg

HDR_FIXED = 4 + 8  # version + strlen uint64 (then sensor name variable)

def extract_scene(scene: str, needed_frames: set[int]) -> None:
    sens_path = f'{ROOT_IN}/{scene}/{scene}.sens'
    out_dir = f'{ROOT_OUT}/{scene}/sens'
    os.makedirs(out_dir, exist_ok=True)

    with open(sens_path, 'rb') as f:
        version = struct.unpack('I', f.read(4))[0]
        assert version == 4, f'unexpected version {version}'
        strlen = struct.unpack('Q', f.read(8))[0]
        f.seek(strlen, 1)                                  # sensor_name
        f.seek(16*4 * 4, 1)                                # 4 × (16 floats)
        f.seek(4 + 4, 1)                                   # color/depth compression enum
        color_w, color_h, depth_w, depth_h = struct.unpack('IIII', f.read(16))
        f.seek(4, 1)                                       # depth_shift
        num_frames = struct.unpack('Q', f.read(8))[0]

        remaining = set(needed_frames)
        written = 0
        for idx in range(num_frames):
            if not remaining:
                break
            # per-frame header: 64 (cam_to_world) + 8 + 8 + 8 + 8
            f.seek(16*4, 1)
            ts_c, ts_d = struct.unpack('QQ', f.read(16))
            color_size = struct.unpack('Q', f.read(8))[0]
            depth_size = struct.unpack('Q', f.read(8))[0]
            if idx in remaining:
                data = f.read(color_size)
                with open(f'{out_dir}/frame-{idx:06d}.color.jpg', 'wb') as g:
                    g.write(data)
                f.seek(depth_size, 1)
                remaining.discard(idx)
                written += 1
            else:
                f.seek(color_size + depth_size, 1)
        missing = sorted(remaining)
    print(f'  {scene}: wrote {written}/{len(needed_frames)}; num_frames={num_frames}; '
          f'color={color_w}x{color_h}; missing={missing}')


def main():
    for scene in sorted(NEEDED):
        extract_scene(scene, set(NEEDED[scene]))


if __name__ == '__main__':
    main()
