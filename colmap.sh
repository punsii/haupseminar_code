SCENE=../data/scene2

colmap automatic_reconstructor \
  --image_path $SCENE/images --workspace_path $SCENE \
  --data_type individual --single_camera 1 --camera_model OPENCV \
  --dense 0 --use_gpu 0

colmap image_undistorter \
  --image_path $SCENE/images --input_path $SCENE/sparse/0 \
  --output_path $SCENE/undistorted --output_type COLMAP

colmap model_analyzer --path $SCENE/undistorted/sparse

colmap model_converter \
  --input_path $SCENE/undistorted/sparse \
  --output_path $SCENE/undistorted/sparse --output_type TXT

