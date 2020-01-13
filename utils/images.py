#finds all .png images in folder, and resize them using cv2
def resize_images_in_folder(in_dir, out_dir, im_size):
    filenames = glob.glob(os.path.join(in_dir, "*.png"))
    for fname in filenames:
        print(fname)
        file_name = os.path.basename(fname)
        img = cv2.imread(fname)
        resized = cv2.resize(img, (im_size, im_size),  interpolation = cv2.INTER_AREA )
        cv2.imwrite(os.path.join(out_dir, file_name), resized)

#finds all .png images in folder, and grayscale them using cv2
def grayscale_images_in_folder(in_dir, out_dir, im_size):
    filenames = glob.glob(os.path.join(in_dir, "*.png"))
    for fname in filenames:
        print(fname)
        file_name = os.path.basename(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(out_dir, file_name), gray)