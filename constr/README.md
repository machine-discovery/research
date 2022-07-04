# Unifying physical systems’ inductive biases in neural ODE using dynamics constraints

[Paper link] https://openreview.net/forum?id=IdhNZeIE7i7

The code should be compatible with `python>=3.7`. Please raise issues if you run into any problems.

To install the packages required:

    conda install python==3.7
    pip install torch torchvision torchaudio
    pip install pytorch-lightning==1.6.3 numpy==1.21.5 gym==0.23.1 functorch==0.1.1 pickle5
    pip install git+https://github.com/VLL-HD/FrEIA.git


To replicate the experiments in the paper, follow the following commands assuming gpu is used. Some of the omitted 
parameters are assumed to be default per `utils.py`.

Task 1 \
`python main.py --version 1 --experiment mass-spring --model baseline` \
`python main.py --version 2 --experiment mass-spring --model hnn` \
`python main.py --version 3 --experiment mass-spring --model lnn` \
`python main.py --version 4 --experiment mass-spring --model nsf` \
`python main.py --version 5 --experiment mass-spring --model baselinereg --reg_real 100000`

Task 2\
`python main.py --version 11 --experiment single-pdl --model baseline`\
`python main.py --version 12 --experiment single-pdl --model hnn`\
`python main.py --version 13 --experiment single-pdl --model lnn`\
`python main.py --version 14 --experiment single-pdl --model nsf`\
`python main.py --version 15 --experiment single-pdl --model baselinereg --reg_real 10000`

Task 3\
`python main.py --version 21 --experiment double-pdl --model baseline --batch 10000`\
`python main.py --version 22 --experiment double-pdl --model hnn --batch 10000`\
`python main.py --version 23 --experiment double-pdl --model lnn --batch 10000`\
`python main.py --version 24 --experiment double-pdl --model nsf --batch 10000`\
`python main.py --version 25 --experiment double-pdl --model invertiblennreg --reg_real 1000 --batch 10000`

Task 4\
`python main.py --version 31 --experiment damped-single-pdl --model damped-baseline`\
`python main.py --version 32 --experiment damped-single-pdl --model dampedregbaseline --reg_real 100`

Pixel-pendulum\
`python main.py --version 41 --experiment pixel-pdl --model pixelhnn`\
`python main.py --version 42 --experiment pixel-pdl --model pixelreg --reg_real 0`\
`python main.py --version 43 --experiment pixel-pdl --model pixelreg --reg_real 500`

Damped pixel-pendulum\
`python main.py --version 41 --experiment damped-pixel-pdl --model dampedpixelreg --reg_real 0`\
`python main.py --version 42 --experiment damped-pixel-pdl --model dampedpixelreg --reg_real 500`

