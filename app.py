import os
import cv2
import gradio as gr
import AnimeGANv3_src


os.makedirs('output', exist_ok=True)


def inference(img_path, Style, if_face=None):
    print(img_path, Style, if_face)
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if Style == "AnimeGANv3_Arcane":
            f = "A"
        elif Style == "AnimeGANv3_Trump v1.0":
            f = "T"
        elif Style == "AnimeGANv3_Shinkai":
            f = "S"
        elif Style == "AnimeGANv3_PortraitSketch":
            f = "P"
        elif Style == "AnimeGANv3_Hayao":
            f = "H"
        elif Style == "AnimeGANv3_Disney v1.0":
            f = "D"
        elif Style == "AnimeGANv3_JP_face v1.0":
            f = "J"
        elif Style == "AnimeGANv3_Kpop v2.0":
            f = "K"
        else:
            f = "U"

        try:
            det_face = True if if_face=="Yes" else False
            output = AnimeGANv3_src.Convert(img, f, det_face)
            save_path = f"output/out.{img_path.rsplit('.')[-1]}"
            cv2.imwrite(save_path, output[:, :, ::-1])
            return output, save_path
        except RuntimeError as error:
            print('Error', error)
    except Exception as error:
        print('global exception', error)
        return None, None


title = "AnimeGANv3: To produce your own animation."
description = r"""Official online demo for <a href='https://github.com/TachibanaYoshino/AnimeGANv3' target='_blank'><b>AnimeGANv3</b></a>. If you like what I'm doing you can tip me on <a href='https://www.patreon.com/Asher_Chan' target='_blank'><b>**patreon**</b></a>.<br> 
It can be used to turn your photos or videos into anime.<br>
To use it, simply upload your image. It can convert landscape photos to Hayao Miyazaki or Makoto Shinkai style anime, as well as 4 style conversions about human faces.<br>
If AnimeGANv3 is helpful, please help to ⭐ the <a href='https://github.com/TachibanaYoshino/AnimeGANv3' target='_blank'>Github Repo</a> and recommend it to your friends. 😊

"""
article = r"""

[![GitHub Stars](https://img.shields.io/github/stars/TachibanaYoshino/AnimeGANv3?style=social)](https://github.com/TachibanaYoshino/AnimeGANv3)    

### 🗻 Demo
I. Video to anime (Hayao Style)   
<p style="display: flex;">
  <a href="https://youtu.be/EosubeJmAnE" target="___blank" style="margin-left: 14px;"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 1&color=red"/></a>   
  <a href="https://youtu.be/5qLUflWb45E" target="___blank" style="margin-left: 14px;"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 2&color=green"/></a>   
  <a href="https://youtu.be/0KaScDxgyBw" target="___blank" style="margin-left: 14px;"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 3&color=pink"/></a>   
</p>
II. Video to anime (USA cartoon + Disney style ) 
<a href="https://youtu.be/vJqQQMRYKh0"><img src="https://img.shields.io/static/v1?label=YouTube&message=AnimeGANv3_Trump style v1.5 &color=gold"/></a>   

----------

## License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications. Permission is granted to use the AnimeGANv3 given that you agree to my license terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.   

## Acknowledgement 
* Huggingface UI is referenced from @akhaliq/GFPGAN.     
* The dataset of AnimeGANv3_JP_face v1.0 is from DCTnet and then manually optimized.       

## Author  
Xin Chen    
If you have any question, please open an issue on GitHub Repo.    
         

<center><img src='https://visitor-badge.glitch.me/badge?page_id=AnimeGANv3_online' alt='visitor badge'></center>
"""
gr.Interface(
    inference, [
        gr.inputs.Image(type="filepath", label="Input"),
        gr.Dropdown([
            'AnimeGANv3_Hayao',
            'AnimeGANv3_Shinkai',
            'AnimeGANv3_Arcane',
            'AnimeGANv3_USA',
            'AnimeGANv3_Trump v1.0',
            'AnimeGANv3_Disney v1.0',
            'AnimeGANv3_PortraitSketch',
            'AnimeGANv3_JP_face v1.0',
            'AnimeGANv3_Kpop v2.0',
        ],
            type="value",
            value='AnimeGANv3_Hayao',
            label='AnimeGANv3 Style'),
        gr.inputs.Radio(['Yes', 'No'], type="value", default='No', label='Extract face'),
    ], [
        gr.outputs.Image(type="numpy", label="Output (The whole image)"),
        gr.outputs.File(label="Download the output image")
    ],
    title=title,
    description=description,
    article=article,
    allow_flagging="never",
    examples=[['samples/7_out.jpg', 'AnimeGANv3_Arcane', "Yes"],['samples/1_out.jpg', 'AnimeGANv3_Kpop v2.0', "Yes"], ['samples/15566.jpg', 'AnimeGANv3_USA', "Yes"],['samples/23034.jpg', 'AnimeGANv3_Trump v1.0', "Yes"], ['samples/jp_13.jpg', 'AnimeGANv3_Hayao', "No"],
              ['samples/jp_20.jpg', 'AnimeGANv3_Shinkai', "No"], ['samples/Hamabe Minami.jpg', 'AnimeGANv3_Disney v1.0', "Yes"], ['samples/120.jpg', 'AnimeGANv3_JP_face v1.0', "Yes"], ['samples/52014.jpg', 'AnimeGANv3_PortraitSketch', "Yes"]]).launch(enable_queue=True)
