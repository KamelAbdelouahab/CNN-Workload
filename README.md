# CNN Workload Benchmark
A script to evaluate the computational workload of popular CNN models winning ILSVRC.

To use : 
`python benchmarkModel.py <.prototxt> <.caffemodel>`

Please be sure to download the models using the appropriate script

# Results
|         Model         | AlexNet | GoogleNet |  VGG16 |  VGG19 | ResNet50 | ResNet101 | ResNet-152 |
|:---------------------:|:-------:|:---------:|:------:|:------:|:--------:|:---------:|:----------:|
|        Top1 err       |  42.9 % |   31.3 %  | 28.1 % | 27.3 % |   24.7%  |  23.6% %  |    23.0%   |
|        Top5 err       | 19.80 % |  10.07 %  | 9.90 % | 9.00 % |   7.8 %  |   7.1 %   |    6.7 %   |
|      conv layers      |    5    |     57    |   13   |   16   |    53    |    104    |     155    |
|  conv workload (MACs) |  666 M  |   1.58 G  | 15.3 G | 19.5 G |  3.86 G  |   7.57 G  |   11.3 G   |
|    conv parameters    |  2.33 M |   5.97 M  | 14.7 M |  20 M  |  23.5 M  |   42.4 M  |    58 M    |
|      pool layers      |    3    |     14    |    5   |    5   |     2    |     2     |      2     |
|       FC layers       |    3    |     1     |    3   |    3   |     1    |     1     |      1     |
|   FC workload (MACs)  |  58.6 M |   1.02 M  |  124 M |  124 M |  2.05 M  |   2.05 M  |   2.05 M   |
|      FC parametrs     |  58.6 M |   1.02 M  |  124 M |  124 M |  2.05 M  |   2.05 M  |   2.05 M   |
| Total workload (MACs) |  724 M  |   1.58 G  | 15.5 G | 19.6 G |  3.86 G  |   7.57 G  |   11.3 G   |
|    Total parameters   |   61 M  |   6.99 M  |  138 M |  144 M |  25.5 M  |   44.4 M  |    60 M    |
