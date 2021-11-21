from infer import InferenceHelper

infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a batched rgb tensor
# example_rgb_batch = ...  
# bin_centers, predicted_depth = infer_helper.predict(example_rgb_batch)

# predict depth of a single pillow image
# img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
# bin_centers, predicted_depth = infer_helper.predict_pil(img)

# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
infer_helper.predict_dir("/data/xyao/sg_dataset/sg_train_images/", "/data/xyao/adabins_depth/depth_results/train/")