import os

from features.feat_SIFT import FeatSIFT


def main(img, desc, save_img, img_out, ext):
    if os.path.isfile(img):
        feat = FeatSIFT(
            filepath=img, desc_out=desc, save_img_out=save_img, img_out=img_out, ext=ext
        )
        print(f"Feature extraction for {img} using {feat._feature_name}")
        d_out, i_out = feat.pipeline()
        print(f"Descriptors saved to {d_out}")
        print(f"Descriptors shape: {feat.desc.shape}")
        if save_img:
            print(f"Image saved to {i_out}\n")
    elif os.path.isdir(img):
        for file in os.listdir(img):
            filepath = os.path.join(img, file)
            if os.path.isfile(filepath) and (
                file.endswith(".jpg") or file.endswith(".png")
            ):
                feat = FeatSIFT(
                    filepath=filepath,
                    desc_out=desc,
                    ext=ext,
                    save_img_out=save_img,
                    img_out=img_out,
                )
                print(f"Feature extraction for {filepath} using {feat._feature_name}")
                d_out, i_out = feat.pipeline()
                print(f"Descriptors saved to {d_out}")
                print(f"Descriptors shape: {feat.desc.shape}")
                if save_img:
                    print(f"Image saved to {i_out}\n")
    else:
        raise ValueError("img must be a file or a directory!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="image file path or directory containing images")
    parser.add_argument("desc", help="descriptor file path")
    parser.add_argument("--save_img", help="save image", action="store_true")
    parser.add_argument(
        "--img_out",
        help="image output file path",
        default=os.path.join(os.getcwd(), "output"),
    )
    parser.add_argument("--ext", help="descriptor file extension", default="npy")
    args = parser.parse_args()
    main(args.img, args.desc, args.save_img, args.img_out, args.ext)
