from torchvision import datasets
from torchvision import transforms
import torch
import vgg
import glob
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pad

model = vgg.__dict__['vgg19']()
model.features = torch.nn.DataParallel(model.features)

# loading pretrained model from github
checkpoint = torch.load('model_vgg19_github.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

# # loading pretrained model from github
# checkpoint = torch.load('model_vgg19_pytorch.pth') # this is for 1000 classes
# model.load_state_dict(checkpoint)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

resize = transforms.Resize(32)

# official cifar10 dataset
# images = datasets.CIFAR10(".", download=True)

images = []
filenames = []
for f in glob.iglob("./image_data/bird_images/*"):
    images.append((Image.open(f)))
    filenames.append(f)

labels = np.zeros(len(images))

for image, label, filename in zip(images, labels, filenames):

    # image.show()

    # todo: pad or resize
    # --- resize mode ---
    # image = transforms.ToTensor()(resize(image))
    # image = normalize(image).unsqueeze_(0)

    # --- pad mode ---
    image = transforms.ToTensor()(pad(image, (3, 3, 4, 4), padding_mode='edge'))
    image = normalize(image).unsqueeze_(0)

    if torch.cuda.is_available():
        input_batch = image.to('cuda')
        model.to('cuda')

        with torch.no_grad():
            _output = model(input_batch)
            m = torch.nn.Softmax(dim=1)
            output = m(_output)

    print(filename)
    print('prediction: class ' + str(np.argmax(output.cpu().numpy(), axis=1)) + ' with ' + 'probability ' + str(np.max(output.cpu().numpy())))
    print("\n")
