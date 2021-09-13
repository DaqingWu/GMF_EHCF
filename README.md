# Environment
+ python[3.7.4]
+ torch[1.8.0]
+ numpy[1.17.2]
+ pandas[0.25.1]

# To run code
+ preprocess dataset
  + python preprocess.py --name Beibei
+ run GMF on the Beibei dataset
  + python main.py --name Beibei --module GMF --gpu 0
+ run EHCF on the Beibei dataset
  + python main.py --name Beibei --module EHCF --gpu 1
