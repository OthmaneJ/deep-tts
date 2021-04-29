import argparse, sys, os
import torch
# from librosa.output import write_wav

sys.path.append('./speech')

from speedyspeech import SpeedySpeech
from melgan.model.generator import Generator
from melgan.utils.hparams import HParam
from hparam import HPStft, HPText
from utils.text import TextProcessor
from functional import mask
import math

import dash
import dash_html_components as html
import dash_core_components as dcc
# import plotly.express as px
# from IPython.display import Audio
# from IPython.utils import io
# from synthesizer.inference import Synthesizer
# from encoder import inference as encoder
# from vocoder import inference as vocoder
from pathlib import Path
# import numpy as np
import scipy
import soundfile as sf
import json
# import time

# from librosa.output import write_wav

speedyspeech_checkpoint = 'checkpoints/speedyspeech.pth'
melgan_checkpoint = 'checkpoints/melgan.pth'
device = 'cpu'
audio_folder = 'assets'
input_text = "Get a complete visualization of your app in a team-based continuous delivery environment"


# with open('latest_embeddings.json') as f:
#   new_embeddings = json.load(f)
  
# celebrities = [el['name'] for el in new_embeddings]

# encoder_weights = Path("./encoder/saved_models/pretrained.pt")
# vocoder_weights = Path("./vocoder/saved_models/pretrained.pt")
# syn_dir = Path("./synthesizer/saved_models/pretrained/pretrained.pt")
# encoder.load_model(encoder_weights)
# synthesizer = Synthesizer(syn_dir)
# vocoder.load_model(vocoder_weights)


external_stylesheets = [
    "https://use.fontawesome.com/releases/v5.0.7/css/all.css",
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
    'https://fonts.googleapis.com/css?family=Roboto&display=swap'
]

# app = dash.Dash(__name__)

app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)

server = app.server

app.layout = html.Div(
    [
        html.H4(children="Fast and Light Text-to-Speech"),
        # dcc.Markdown("Clone the voice of your favourite celebrity using Deep Learning."),
        dcc.Markdown("""
    **Instructions:** type a sentence (between 10 and 20 words), then click submit and wait for about 10 seconds
    """,
        ),
        # dcc.Markdown("**Choose your celebrity**"),
        # html.Div(html.Img(id='celebrity_img',src='https://m.media-amazon.com/images/M/MV5BMTc1MDI0MDg1NV5BMl5BanBnXkFtZTgwMDM3OTAzMTE@._V1_SY1000_CR0,0,692,1000_AL_.jpg',style={'width':'200px'}),style={'marginTop':'10px',"marginBottom":'10px'}),
        # dcc.Dropdown(id="celebrity-dropdown",options=[{'label':celebrity,'value':i} for i,celebrity in enumerate(celebrities)]),
        # html.Div(id="slider-output-container"),
        dcc.Markdown("**Type a sentence and click submit**"),
        html.Div(dcc.Textarea(id="transcription_input",maxLength=300,rows=2,style={'width':'100%'},
                              value ='I believe in living in the present and making each day count. I don’t pay much attention to the past or the future.')),
        html.Div(html.Button('Submit', id='submit', n_clicks=0)),
        html.Br(),
        dcc.Loading(id="loading-1",
                    children=[html.Audio(id="player",src = "assets/0.wav", controls=True, style={
          "width": "100%",
        })],type='default'),
        html.H4('How would you rate the quality of the audio ?'),
        dcc.Slider(id='rating',max=5,min=1,step=1,marks={i: f'{i}' for i in range(1, 6)},),
        # dcc.Graph(id="waveform", figure=fig),
        html.Div(html.Button('Rate', id='rate-button', n_clicks=0)),
        html.H4("Please put a rating up here!",id='rating-message'),
        dcc.ConfirmDialog(id='confirm',message="Too many words (>50) or too little (<10) may effect the quality of the audio, continue at your own risk ^^'"),
        # html.A(children=[html.Img(src='https://cdn.buymeacoffee.com/buttons/default-orange.png',alt="Buy Me Coffee",height="41",width="174")],href='https://www.buymeacoffee.com/OthmaneJ'),
    
    ]
    ,style={'textAlign': 'center','marginRight':'100px','marginLeft':'100px','marginTop':'50px','marginBottom':'50px'})

# # Set picture of celebrity
# @app.callback(
#     dash.dependencies.Output('celebrity_img','src'),
#     [dash.dependencies.Input('celebrity-dropdown','value')]
# )

# def display_image(celebrity):
#   return new_embeddings[celebrity]['img']


# Transcribe audio
@app.callback(
    dash.dependencies.Output("confirm", "displayed"),
    [dash.dependencies.Input("submit","n_clicks")],
    # [dash.dependencies.State("celebrity-dropdown","value"),
    [dash.dependencies.State("transcription_input", "value")],
)

def display_warning(n_clicks,value):
    n_words=  len(value.split(' '))
    print(n_words)
    if n_words>50 or n_words<10:
        return True
    return False

#  Transcribe audio
@app.callback(
    dash.dependencies.Output("player", "src"),
    [dash.dependencies.Input("submit","n_clicks"),
     ],
    # [dash.dependencies.State("celebrity-dropdown","value"),
    [dash.dependencies.State("transcription_input", "value")],
)

def vocalize(n_clicks,value):
    
    input_text = value

    speedyspeech_checkpoint = 'checkpoints/speedyspeech.pth'
    melgan_checkpoint = 'checkpoints/melgan.pth'
    device = 'cpu'
    audio_folder = 'assets'
    input_text = "Get a complete visualization of your app in a team-based continuous delivery environment"

    print('Loading model checkpoints')
    m = SpeedySpeech(
        device=device
    ).load(speedyspeech_checkpoint, map_location=device)
    m.eval()

    checkpoint = torch.load(melgan_checkpoint, map_location=device)
    hp = HParam("speech/melgan/config/default.yaml")
    melgan = Generator(hp.audio.n_mel_channels).to(device)
    melgan.load_state_dict(checkpoint["model_g"])
    melgan.eval(inference=False)

    with open('./counter.txt','r') as f:
        counter = int(f.read())
    print(f'number of files : {counter}')
    
    print('Processing text')
    txt_processor = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
    text = [input_text]

    phonemes, plen = txt_processor(text)
    # append more zeros - avoid cutoff at the end of the largest sequence
    phonemes = torch.cat((phonemes, torch.zeros(len(phonemes), 5).long() ), dim=-1)
    phonemes = phonemes.to(device)

    print('Synthesizing')
    # generate spectrograms
    with torch.no_grad():
        spec, durations = m((phonemes, plen))


    # invert to log(mel-spectrogram)
    spec = m.collate.norm.inverse(spec)

    # mask with pad value expected by MelGan
    msk = mask(spec.shape, durations.sum(dim=-1).long(), dim=1).to(device)
    spec = spec.masked_fill(~msk, -11.5129)

    # Append more pad frames to improve end of the longest sequence
    spec = torch.cat((spec.transpose(2,1), -11.5129*torch.ones(len(spec), HPStft.n_mel, 5).to(device)), dim=-1)

    # generate audio
    with torch.no_grad():
        audio = melgan(spec).squeeze(1)

    print('Saving audio')
    # TODO: cut audios to proper length
    audio_folder = '/assets'
    # for i,a in enumerate(audio.detach().cpu().numpy()):
        # write_wav(os.path.join(audio_folder,f'{i}.wav'), a, HPStft.sample_rate, norm=False)
    sf.write(os.path.join(audio_folder,f'{counter}.wav'), audio.detach().cpu().numpy()[0], HPStft.sample_rate)
    counter +=1
    # time.sleep(10)
    with open('./counter.txt','w') as f:
        f.write(str(counter))

    return f'/assets/{counter-1}.wav'

@app.callback(
    dash.dependencies.Output("rating-message", "value"),
    [dash.dependencies.Input("rate-button","n_clicks"),
     ],
    [dash.dependencies.State("rating","value")], 
)

def print_rating(n_clicks,rating):
    print(rating)
    return 'your rating is ' + str(rating)


if __name__ == "__main__":

    # print('Loading model checkpoints')
    # m = SpeedySpeech(
    #     device=device
    # ).load(speedyspeech_checkpoint, map_location=device)
    # m.eval()

    # checkpoint = torch.load(melgan_checkpoint, map_location=device)
    # hp = HParam("speech/melgan/config/default.yaml")
    # melgan = Generator(hp.audio.n_mel_channels).to(device)
    # melgan.load_state_dict(checkpoint["model_g"])
    # melgan.eval(inference=False)

    app.run_server(debug=False)
