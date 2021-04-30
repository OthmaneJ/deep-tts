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

from pathlib import Path
import scipy
import soundfile as sf
import json

from io import BytesIO
import base64

from scipy.io.wavfile import write
import numpy as np


speedyspeech_checkpoint = './checkpoints/speedyspeech.pth'
melgan_checkpoint = './checkpoints/melgan.pth'
device = 'cpu'
audio_folder = 'assets'
input_text = "Get a complete visualization of your app in a team-based continuous delivery environment"

external_stylesheets = [
    "https://use.fontawesome.com/releases/v5.0.7/css/all.css",
    'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
    'https://fonts.googleapis.com/css?family=Roboto&display=swap'
]

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


class SpeedySpeechInference:
    def __init__(self, speedyspeech_checkpoint, melgan_checkpoint, device):
        self.device = device
        self.speedyspeech = self._setup_speedyspeech(speedyspeech_checkpoint)
        self.melgan = self._setup_melgan(melgan_checkpoint)
        self.txt_processor = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
        self.bit_depth = 16
        self.sample_rate = 22050

    def _setup_speedyspeech(self, checkpoint):
        speedyspeech = SpeedySpeech(
            device=self.device
        ).load(checkpoint, map_location=self.device)
        speedyspeech.eval()
        return speedyspeech

    def _setup_melgan(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        hp = HParam("./speech/melgan/config/default.yaml")
        melgan = Generator(hp.audio.n_mel_channels).to(self.device)
        melgan.load_state_dict(checkpoint["model_g"])
        melgan.eval(inference=False)
        return melgan

    def synthesize(self, input_text):
        print(input_text)
        text = [input_text.strip()]
        phonemes, plen = self.txt_processor(text)

        # append more zeros - avoid cutoff at the end of the largest sequence
        phonemes = torch.cat((phonemes, torch.zeros(len(phonemes), 5).long() ), dim=-1)
        phonemes = phonemes.to(self.device)

        # generate spectrograms
        with torch.no_grad():
            spec, durations = self.speedyspeech((phonemes, plen))

        # invert to log(mel-spectrogram)
        spec = self.speedyspeech.collate.norm.inverse(spec)

        # mask with pad value expected by MelGan
        msk = mask(spec.shape, durations.sum(dim=-1).long(), dim=1).to(self.device)
        spec = spec.masked_fill(~msk, -11.5129)

        # Append more pad frames to improve end of the longest sequence
        spec = torch.cat((
            spec.transpose(2,1),
            -11.5129 * torch.ones(len(spec), HPStft.n_mel, 5).to(self.device)
        ), dim=-1)

        # generate audio
        with torch.no_grad():
            audio = self.melgan(spec).squeeze(1)
            audio = audio.detach().cpu().numpy()[0]

        # denormalize
        x = 2 ** self.bit_depth - 1
        audio = np.int16(audio * x)
        return audio


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
    [dash.dependencies.State("transcription_input", "value")],
)

def vocalize(n_clicks,value):
    
    print('Saving audio')

    speedyspeech = SpeedySpeechInference(
        speedyspeech_checkpoint,
        melgan_checkpoint,
        device
    )
    
    input_text = value
    buf = BytesIO()
    waveform_integers = speedyspeech.synthesize(input_text)
    write(buf, speedyspeech.sample_rate, waveform_integers)

    return f'data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}'

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

    app.run_server(debug=False)
