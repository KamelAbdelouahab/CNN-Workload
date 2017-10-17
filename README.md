# paramAnalyzer
A script to analyse popular CNN models and their computational load
To use : 
`python cnnAnalyzer.py <.prototxt> <.caffemodel>`
Please be sure to download the models using the appropriate script

# Results
|       Model      | AlexNet | GoogleNet |  VGG16 |  VGG19 | ResNet50 | ResNet100 | SqueezNet |
|:----------------:|:-------:|:---------:|:------:|:------:|:--------:|:---------:|:---------:|
|     Top1 err     |         |           |        |        |          |           |           |
|     Top5 err     |         |           |        |        |          |           |           |
|    conv layers   |    5    |     57    |   13   |   16   |    53    |    104    |     26    |
|  conv workload   |  666 M  |   1.58 G  | 15.3 G | 19.5 G |  3.86 G  |   7.57 G  |   861 M   |
|  conv parameters |  2.33 M |   5.97 M  | 14.7 M |  20 M  |  23.5 M  |   42.4 M  |   1.24 M  |
|    pool layers   |    3    |     14    |    5   |    5   |     2    |     2     |     4     |
|     FC layers    |    3    |     1     |    3   |    3   |     1    |     1     |     0     |
|    FC workload   |  58.6 M |   1.02 M  |  124 M |  124 M |  2.05 M  |   2.05 M  |     0     |
|   FC parametrs   |  58.6 M |   1.02 M  |  124 M |  124 M |  2.05 M  |   2.05 M  |     0     |
|  Total workload  |  724 M  |   1.58 G  | 15.5 G | 19.6 G |   3.86   |   7.57 G  |   861 M   |
| Total parameters |   61 M  |   6.99 M  |  138 M |  144 M |  25.5 M  |   44.4 M  |   1.24 M  |

