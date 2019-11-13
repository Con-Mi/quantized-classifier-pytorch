def load_jit_model(_MODEL_JIT_FILENAME_):
    r'''
        The function takes the filename of a model that has been exported
        into the jit format and is loaded to torch so that it can be used
        in the development.
        Args:
            @param      _MODEL_JIT_FILENAME_:  The filename of the .pt file

            @return:    model:  Returns the model loaded and in evaluation mode.

        Example:
            >>> model = load_jit_model("../jit_models_bin/mnasnet0_5.pt")
    '''
    import torch


    model = torch.jit.load(_MODEL_JIT_FILENAME_)

    return model.eval()


def load_cv_image(_IMAGE_FILENAME_):
    r'''
        The function loads an image using OpenCV image reading function and
        converts it to RGB and then gets resized to ( H = 224, W = 224 ).

        Args:
            @param _IMAGE_FILENAME_: The filename of the image that needs to be loaded to the
                                        application

            @return image:            Returns the image loaded as an OpenCV Mat constructor.

        Example:
            >>> image = load_cv_image(_IMAGE_FILENAME_)
            # image is now an OpenCV Mat in RGB with dimensions H = 224, W = 224
    '''
    import cv2


    image = cv2.imread( _IMAGE_FILENAME_ )
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    image = cv2.resize( image, (224, 224) )
    return image


def load_ocv_image_as_tensor( ocv_image ):
    r'''
        The function takes an image and creates
        a PyTorch tensor format. The loaded tensor is then normalized as
        it is normalized on ImageNet.

        Args:
            @param ocv_image:    The input is an OpenCV Mat constructor from an Image filename.

            @return tnsr_image:  Returns a PyTorch Tensor

        Example:
            >>> image = load_cv_image(_IMAGE_FILENAME_)
            >>> tnsr_image = load_ocv_image_as_tensor( image )
            # tnsr_image is now a PyTorch tensor image that has been normalized with
            # dimensions (B, C, H, W) = (1, 3, 224, 224)
    '''
    import torchvision

    trsfm = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
    ])
    tnsr_image = trsfm( ocv_image )
    tnsr_image = tnsr_image.unsqueeze(dim=0)

    return tnsr_image


def jit_inference( jit_model, image_tnsr ):
    r'''
        Inference function using a jit model on a PyTorch image tensor.
        It returns the highest probability value form the 1000 output and the
        corresponding index position.

        Args:
            @param jit_model:   The model loaded from a .pt file exported in with jit.
            @param image_tnsr:  The image loaded as a PyTorch tensor and normalized.

            @return prob_value:         The max probability value found.
            @return imagenet_index:     The corresponding index of the max probability value.

    '''
    import torch


    output = jit_model(image)
    output = torch.softmax(output, 1)
    prob_value, imagenet_index = torch.max(output, 1)

    return prob_value, imagenet_index