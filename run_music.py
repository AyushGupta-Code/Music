from map_emo_to_music import map_emotions_to_music
from para_to_emo import detect_emotion
from audiocraft.models import MusicGen
import soundfile as sf
#import torch  # not needed here if detect_emotion handles it
#from audiocraft.utils.notebook import display_audio  # optional for Jupyter

def generate_music(prompt):
    model = MusicGen.get_pretrained('facebook/musicgen-melody')  # Use full pretrained ID
    model.set_generation_params(duration=10)  # seconds
    wav = model.generate([prompt])  # wav shape: [batch_size, channels, samples]

    audio = wav[0].cpu().numpy()  # get first sample in batch

    # If stereo or multi-channel, transpose to (samples, channels)
    if audio.ndim == 2:  # (channels, samples)
        audio = audio.T  # to (samples, channels)

    sf.write('output.wav', audio, samplerate=32000)
    print("🎵 Music saved to output.wav")

if __name__ == "__main__":
    paragraph = """As the enemy forces lay siege to Mahishmati, Amarendra Baahubali fights fiercely on the battlefield, cutting down soldiers with unmatched skill. The battle rages violently, with explosions and clashing swords all around. Suddenly, Baahubali spots the injured queen mother, Sivagami, struggling in the chaos. Without hesitation, he rushes to her side and lifts her onto his back, determined to save her life. What follows is an epic and breathtaking sequence where Baahubali climbs the steep, slippery rocks beside the roaring Kuntala waterfall while carrying Sivagami. The camera captures every moment of his struggle — muscles straining, water crashing down — as he pushes himself to the limits. Just as he nears the top, Bhallaladeva’s archers fire arrows at him, and Bhallaladeva himself watches coldly from below. Baahubali’s path is blocked by Bhallaladeva’s soldiers, and a brutal fight ensues. Despite Baahubali’s heroic effort, the scene cuts to black with an ambiguous shot of him surrounded, leaving the audience breathless and hanging on his fate. The waterfall, the fierce battle, and the close-up on Baahubali’s determined face make this climax a powerful blend of action, emotion, and suspense."""

    emotions = detect_emotion(paragraph)
    music_profile = map_emotions_to_music(emotions)
    
    if "prompt" in music_profile:
        generate_music(music_profile["prompt"])
    else:
        print("Error: 'prompt' key not found in music profile.")
