import cv2,os
import AnimeGANv3_src
if __name__ == '__main__':

    f = 'A'
    input_imgs_path = r'../../v3-usa\dataset\USA\val'
    # input_imgs_path = r'/mnt/data/xinchen/v3-usa/dataset/USA/val'
    output_path = 'AnimeGANv3_usa_64_output'
    # img = cv2.imread(os.path.join(input_imgs_path, os.listdir(input_imgs_path)[0]))
    img = cv2.imread(os.path.join(input_imgs_path, 'jp_16.png'))
    out = AnimeGANv3_src.Convert(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f, True)
    # cv2.imshow('d', cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
    cv2.imwrite('a.jpg', cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


