import os
import whisper
import argparse
from whisper.utils import str2bool, get_writer
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=str,
                        help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=[
                        "txt", "vtt", "srt", "tsv", "json", "all"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted(
        [k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="whether to print out the progress and debug messages")
    parser.add_argument("--fp16", type=str2bool, default=True,
                        help="whether to perform inference in fp16; True by default")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")

    print("Loading Whisper model: "+model_name+" ...")

    model = whisper.load_model(model_name)

    writer = get_writer(output_format, output_dir)
    for audio_path in args.pop("audio"):
        print(
            f"Generating subtitles for {audio_path}... This might take a while."
        )
        result = model.transcribe(audio_path, **args,)
        writer(result, os.path.splitext(os.path.basename(audio_path))[0])


if __name__ == "__main__":
    main()
