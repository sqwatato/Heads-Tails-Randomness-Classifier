# Predicting whether a sequence of heads or tails of length 20 was computer generated or human written
Given a sequence of heads or tails of length 20, predict whether it was completely random or human written
### Inspiration
The inspiration for this project was a video of a stats professor that had his students flip a coin 100 times and write down the answer. The professor than looked at each person's sequence of heads and tails and tried guessing if his students cheated or not. He guessed with decent accuracy whiched demonstrated how humans aren't good at faking randomness.
### About
This project contains small implementations of different ai algorithms/models as I learn them. Because this is one of my first ai projects, the code and models aren't exactly the best. I'm still learning and as I learn I would like to add new things to this small project. I ran this project locally on a M1 Macbook Pro, resulting in a different tensorflow library (DON'T USE THE REQUIREMENTS.TXT IF NOT ON M1 MAC)
### Expectations
Because the length of each sequence is only 20 long, it isn't hard for a human not to display any unrandom pattern, likewise any sequence is equally like to happen if randomly generated, so there are going to be cases where a human and computer output the same sequence. Because of that, I don't expect the accuracy to be over 90%, even less 95%.
### Prerequisites
- A device to run code locally or a device to access google colab
### Setup
- Install libraries
    - Normal
        - `pip install matplotlib seaborn numpy pandas tensorflow scikit-learn`
    - M1 Mac (using different versions messes up stuff)
        - `pip install matplotlib seaborn numpy pandas tensorflow-macos==2.10.0 tensorflow-metal==0.6.0 scikit-learn`
    - Google Colab (add code block at beginning of notebook just in case)
        - `!pip install matplotlib seaborn numpy pandas tensorflow scikit-learn`
### Test
Go to [my website](https://jaydenclim.herokuapp.com/) to test out the models without the code!