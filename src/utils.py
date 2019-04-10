import numpy as np
from torchvision import transforms


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


def recover_image(img):
    return (
        (img * np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
               np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) * 255. ).clip(0, 255).astype(np.uint8)


def show_img(img_tensor):
    img = img_tensor.numpy()
    print(img.shape)
    
    img[0] = np.add(img[0] * 0.229, 0.485)
    img[1] = np.add(img[1] * 0.224, 0.456)
    img[2] = np.add(img[2] * 0.225, 0.406)
    
    img = img.transpose(1, 2, 0) * 255.
    img = img.clip(0, 255).astype(np.uint8)
    
    plt.imshow(img)


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

