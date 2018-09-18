# Style image recommendation algorithm

Style image recommendtaion algorithm for a better transferred image. ([slide](https://drive.google.com/open?id=1IpaM344vy2H1Vj5zGp3sgvJRESU3GC2W))
* For quantitative evaluation, we referenced [this paper](https://arxiv.org/abs/1804.00118).


## Usage

### Prerequisites
1. Python packages : keras, numpy, shutil, json, cv2
2. Candidate style image [download](https://drive.google.com/open?id=1Wy_fX94WAgc3o80HE6mP8sI8LMpAqrbY)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Please download the file from link above.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Save the file under `base_style`

3. Saved style image's features [download](https://drive.google.com/open?id=1pr1r_LiDJ4ed8hvxsNVs-RE1e4PjhnZj)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Please download the file from link above.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Save the file under `json_data`

### Running
```
python main.py --content <content file>
```
*Example*:
`python main.py --content content/1.jpg`

#### Arguments
*Required* :  
* `--content`: Filename of the content image

*Optional* :  
* `--top_n`: number of recommended image. *Default*: `5`


## References

#### https://github.com/hwalsuklee/tensorflow-style-transfer
* markdown template
