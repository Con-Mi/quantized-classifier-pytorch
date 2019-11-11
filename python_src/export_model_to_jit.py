def main():
    r'''
        The main loop of the file. Initialize the
        variables and export the model

        Args:
            N/A
        
        Examples:
            >>> main()
    '''
    _MODEL_ = define_pretrained_mnasnet0_5()
    _FILENAME_ = "./jit_models_bin/traced_mnasnet0_5.pt"
    export_to_traced_script(_MODEL_, _FILENAME_)


def define_pretrained_mobilenetV2():
    r'''
        Defines the Mobilenet V2 model with
        pretrained parameters on ImageNet.
        Top 1 Accuracy:     71.88%.
        Top 5 Accuracy:     90.28%

        Inference Latency:  143ms

        Args:
            N/A
        
        Examples:
            >>> model = define_pretrained_mobilenetV2()
    '''
    import torchvision


    return torchvision.models.mobilenet_v2(pretrained=True)


def define_pretrained_mnasnet1_0():
    r'''
        Defines the MnasNet 1_0 model with
        pretrained parameters on ImageNet.
        Top 1 Accuracy:     73.1%.
        Top 5 Accuracy:     91.6%

        Inference Latency:  103ms
        
        Args:
            N/A
        
        Examples:
            >>> model =  define_pretrained_mnasnet1_0()
    '''
    import torchvision


    return torchvision.models.mnasnet1_0(pretrained=True)


def define_pretrained_mnasnet0_5():
    r'''
        Defines the MnasNet 0_5 model with
        pretrained parameters on ImageNet.
        Top 1 Accuracy:     67.4%.
        Top 5 Accuracy:     N/A

        Inference Latency:  83ms
        
        Args:
            N/A
        
        Examples:
            >>> model =  define_pretrained_mnasnet0_5()
    '''
    import torchvision


    return torchvision.models.mnasnet0_5(pretrained=True)


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
    import torch


    dummy_input = torch.randn(1, 3, 224, 224)

    traced_script_module = torch.jit.trace(model.eval(), dummy_input)
    traced_script_module.save(_FILENAME)



if __name__ == "__main__":
    main()