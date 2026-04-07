import argparse
import pathlib
import time
import mujoco as mj
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.params import ROBOT_PR_SPACE_JOINTS_DICT
from rich import print
from tqdm import tqdm
import os
import numpy as np

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["hu_d04"],
        default="hu_d04",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no_viewer",
        action="store_true",
        default=False,
        help="Retarget and save without opening the MuJoCo viewer.",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--save_csv_path",
        default=None,
        help="Optional path to save the robot motion as a flat CSV file.",
    )

    parser.add_argument(
        "--save_pr_csv_path",
        default=None,
        help="Optional path to save robot PR-space joint targets for RL.",
    )

    parser.add_argument(
        "--save_beyondmimic_csv_path",
        default=None,
        help="Optional path to save root pose plus PR-space joints for BeyondMimic.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--max_frames",
        default=None,
        type=int,
        help="Optional maximum number of frames to retarget, useful for quick test exports.",
    )
    
    args = parser.parse_args()

    should_save_motion = (
        args.save_path is not None
        or args.save_csv_path is not None
        or args.save_pr_csv_path is not None
        or args.save_beyondmimic_csv_path is not None
    )

    if should_save_motion:
        for output_path in [
            args.save_path,
            args.save_csv_path,
            args.save_pr_csv_path,
            args.save_beyondmimic_csv_path,
        ]:
            if output_path is None:
                continue
            save_dir = os.path.dirname(output_path)
            if save_dir:  # Only create directory if it's not empty
                os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    if args.max_frames is not None:
        lafan1_data_frames = lafan1_data_frames[:args.max_frames]
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = args.motion_fps
    
    robot_motion_viewer = None
    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                                motion_fps=motion_fps,
                                                transparent_robot=0,
                                                record_video=args.record_video,
                                                video_path=args.video_path,
                                                # video_width=2080,
                                                # video_height=1170
                                                )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)
        

        # visualize
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
                # human_pos_offset=np.array([0.0, 0.0, 0.0])
            )

        if should_save_motion:
            qpos_list.append(qpos.copy())

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
    
    if should_save_motion:
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    if args.save_path is not None:
        import pickle
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    if args.save_csv_path is not None:
        motion = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
        np.savetxt(args.save_csv_path, motion, delimiter=",")
        print(f"Saved CSV to {args.save_csv_path}")

    if args.save_pr_csv_path is not None or args.save_beyondmimic_csv_path is not None:
        pr_joint_names = ROBOT_PR_SPACE_JOINTS_DICT.get(args.robot)
        if pr_joint_names is None:
            raise ValueError(f"Robot {args.robot} does not define a PR-space export order")

        pr_qpos_indices = []
        for joint_name in pr_joint_names:
            joint_id = mj.mj_name2id(retargeter.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint {joint_name} not found in robot model {retargeter.xml_file}")
            pr_qpos_indices.append(int(retargeter.model.jnt_qposadr[joint_id]))

        pr_dof_pos = np.array([[qpos[idx] for idx in pr_qpos_indices] for qpos in qpos_list], dtype=np.float64)

    if args.save_pr_csv_path is not None:
        header = ",".join(pr_joint_names)
        np.savetxt(args.save_pr_csv_path, pr_dof_pos, delimiter=",", header=header, comments="")
        print(f"Saved PR-space CSV to {args.save_pr_csv_path}")
        print(f"PR joint order: {pr_joint_names}")

    if args.save_beyondmimic_csv_path is not None:
        motion = np.concatenate([root_pos, root_rot, pr_dof_pos], axis=1)
        np.savetxt(args.save_beyondmimic_csv_path, motion, delimiter=",")
        print(f"Saved BeyondMimic CSV to {args.save_beyondmimic_csv_path}")
        print(f"BeyondMimic CSV columns: root_pos(3), root_rot_xyzw(4), PR joints({len(pr_joint_names)})")
        print(f"PR joint order: {pr_joint_names}")

    # Close progress bar
    pbar.close()
    
    if robot_motion_viewer is not None:
        robot_motion_viewer.close()
       
