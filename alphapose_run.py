import numpy as np
from common.arguments import parse_args
from common.camera import *
from common.generators import UnchunkedGenerator
from common.loss import *
from common.model import *


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]

def evaluate_alphapose(test_generator, model_pos, action=None, return_predictions=False):
    """
    Inference the 3d positions from 2d position.
    :type test_generator: UnchunkedGenerator
    :param test_generator:
    :param model_pos: 3d pose model
    :param return_predictions: return predictions if true
    :return:
    """
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


def main():

	args = parse_args()	
	metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

	npz = np.load(args.input_npz)
	keypoints = npz['kpts']
	keypoints_symmetry = metadata['keypoints_symmetry']
	kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
	#same with the original: list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
	joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

	# normlization keypoints  Suppose using the camera parameter
	res_w = 1920
	res_h = 1080
	keypoints = normalize_screen_coordinates(keypoints[..., :2], w=res_w, h=res_h)


	#model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
    #                        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
    #                        dense=args.dense)
	model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
	                              dense=args.dense)

	receptive_field = model_pos.receptive_field()
	print('INFO: Receptive field: {} frames'.format(receptive_field))
	pad = (receptive_field - 1) // 2 # Padding on each side
	if args.causal:
	    print('INFO: Using causal convolutions')
	    causal_shift = pad
	else:
	    causal_shift = 0

	model_params = 0
	for parameter in model_pos.parameters():
	    model_params += parameter.numel()
	print('INFO: Trainable parameter count:', model_params)

	if torch.cuda.is_available():
	    model_pos = model_pos.cuda()

	#load model
	chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'])

	#test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
    #                                pad=pad, causal_shift=causal_shift, augment=False,
    #                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
	
	input_keypoints = keypoints.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate_alphapose(gen, model_pos, return_predictions=True)
	print('INFO: Testing on {} frames'.format(test_generator.num_frames()))


	if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
    	#from custom_dataset.py
    	rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
	    prediction = camera_to_world(prediction, R=rot, t=0)

	    # We don't have the trajectory, but at least we can rebase the height
	    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
	    anim_output = {'Reconstruction': prediction}
	    input_keypoints = image_coordinates(input_keypoints[..., :2], w=res_w, h=res_h)

	    #from common.visualization import render_animation
        #render_animation(input_keypoints, keypoints_metadata, anim_output,
        #                 dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
        #                 limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
        #                 input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
        #                 input_video_skip=args.viz_skip)

        from common.visualization import render_animation_alphapose
        #fps 25, azimuth 70
    	render_animation_alphapose(input_keypoints, anim_output,
                     Skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
                     limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                     input_video_path=args.viz_video, viewport=(res_w, res_h),
                     input_video_skip=args.viz_skip)

if __name__ == '__main__':
	main()

