# WebtoonWatermarking
Watermarking images to identify rogue image posters

## Dependencies
- python>=3.6 (strict due to f-string)
- torch
- tqdm

## Usage
Each run.py file support CLI for each model.
Pre-defined run scripts are in exps dir.

Before you run the scripts, you may need following steps.

- Download data from https://drive.google.com/open?id=1EuXrcAlzXQwqb1EDTNh2Xsw3EYJL5ne3 and unzip
- Set environment variables
```
cp env.sh.template env.sh
# edit env.sh
source env.sh  # add it to bashrc to be permanent
```

