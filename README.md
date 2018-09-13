# Style image recommendation algorithm

Style image recommendtaion algorithm for a better transferred image.


## Usage

### Prerequisites
1. Python packages : keras, numpy, shutil, json, cv2
2. Candidate style image [download](https://drive.google.com/open?id=1Wy_fX94WAgc3o80HE6mP8sI8LMpAqrbY)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Please download the file from link above.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Save the file under `base_style`

3. Saved style image's features [download](https://drive.google.com/open?id=1IeWNHhtkYJ5Dq-cjYiD9hPvNHY4B7IO_)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Please download the file from link above.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Save the file under `json_data`

### Running
```
python run_main.py --content <content file>
```
*Example*:
`python run_main.py --content content/1.jpg

#### Arguments
*Required* :  
* `--content`: Filename of the content image

*Optional* :  
* `--top_n`: number of recommended image. *Default*: `5`


## References

#### https://github.com/hwalsuklee/tensorflow-style-transfer
* markdown template
