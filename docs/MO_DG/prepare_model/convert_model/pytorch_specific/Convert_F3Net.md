# Converting a PyTorch F3Net Model {#openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_F3Net}

@sphinxdirective

.. meta::
   :description: Learn how to convert a F3Net model
                 from Pytorch to the OpenVINO Intermediate Representation.


`F3Net <https://github.com/weijun88/F3Net>`__ : Fusion, Feedback and Focus for Salient Object Detection

Cloning the F3Net Repository
############################

To clone the repository, run the following command:

.. code-block:: sh

   git clone http://github.com/weijun88/F3Net.git


Downloading and Converting the Model to ONNX
############################################

To download the pretrained model or train the model yourself, refer to the
`instructions <https://github.com/weijun88/F3Net/blob/master/README.md>`__ in the F3Net model repository. First, convert the model to ONNX format. Create and run the following Python script in the ``src`` directory of the model repository:

.. code-block:: py
   :force:

   import torch
   from dataset import Config
   from net import F3Net

   cfg = Config(mode='test', snapshot=<path_to_checkpoint_dir>)
   net = F3Net(cfg)
   image = torch.zeros([1, 3, 352, 352])
   torch.onnx.export(net, image, 'f3net.onnx', export_params=True, do_constant_folding=True, opset_version=11)


The script generates the ONNX model file ``f3net.onnx``. The model conversion was tested with the commit-SHA: ``eecace3adf1e8946b571a4f4397681252f9dc1b8``.

Converting an ONNX F3Net Model to IR
####################################

.. code-block:: sh

   mo --input_model <MODEL_DIR>/f3net.onnx


@endsphinxdirective
