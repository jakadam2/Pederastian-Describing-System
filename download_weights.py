import gdown
id = '1F8-e18NZlmhMpSCxeTwGh7rwvHhoM2oJ'
id2 = '1VHraNJ-3ZiwFenP1vGptNNxSUZZHA2xQ'

gdown.download(f'https://drive.google.com/uc?/export=download&id={id}',output='./weights/color_model.pt')
gdown.download(f'https://drive.google.com/uc?/export=download&id={id2}',output='./weights/attr_model.pt')
