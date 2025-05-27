How to use those methods:

1. Open your conda/pip environment
2. pip install requirements.txt
3.1 python perturbation_test.py
or 
3.2 python lrp_test.py
or
3.3 python test_layer_norm.py
or 
3.4 python normal_lrp_test.py
or  
3.5 python occlusion/occlusion_test.py

Most classes/scripts have their own main to test the implementation.
Please note that the perturbation_test.py requires at least 12GB of RAM and all your CPU/GPU resources. 
The implentation of LRP for layer normalized LSTMs is in lstm_layer_norm_network.py

For any questions please contact "christoph.wehner@uni-bamberg.de".
