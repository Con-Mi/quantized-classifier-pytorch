import torch
from torchvision import models


def main():
    r'''
        The main loop of the file. Initialize the
        variables and export the model

        Args:
            N/A
        
        Examples:
            >>> main()
    '''
    _MODEL = define_pretrained_mobilenetV2()
    _FILENAME = "./jit_models_bin/traced_mobilenetV2.pt"
    export_to_traced_script(_MODEL, _FILENAME)


def define_pretrained_mobilenetV2():
    r'''
        Defines the Mobilenet V2 model with
        pretrained parameters on Imagenet

        Args:
            N/A
        
        Examples:
            >>> model = define_pretrained_mobilenetV2()
    '''
    return models.mobilenet_v2(pretrained=True)


def export_to_traced_script(model, _FILENAME):

    r'''
        Exports a PyTorch defined model
        to a traced script that can be loaded to
        C++

        Args:
            model:  a model defined in a class module in PyTorch
                    or as a pretrained model from torchvision
            FILENAME:   the name of the file that is going to be
                        saved as. As descriptive as possible.

        Examples:
            >>> export_to_traced_script(squeeze_model, "./jit_models/traced_squeezenet_model.pt") 
    '''
    dummy_input = torch.randn(1, 3, 224, 224)

    traced_script_module = torch.jit.trace(model.eval(), dummy_input)
    traced_script_module.save(_FILENAME)


if __name__ == "__main__":
    main()
