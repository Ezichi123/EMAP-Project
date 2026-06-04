# Understanding the Structure of the EMAP EEG Dataset

This dataset contains **brain activity (EEG), physiological signals, and emotional ratings** recorded while participants watched emotional video clips.

The data has already been processed into **features extracted from short time windows**, so each row represents a **small time segment of the recording**.

Paper with these details linked [here](https://onlinelibrary.wiley.com/doi/10.1111/psyp.14446).
Dataset page linked [here](https://emapchallenge.github.io/).

## Physical Measurement Specs

During the task, electrophysiological activity at the scalp was collected using an active Ag/AgCl 64-electrode net (either actiCAP or actiCAP snap), corresponding to the international extended 10–20 system. Signals were recorded through a BrainVision actiCHamp amplifier running BrainVision Recorder (Brain Products GmbH, www.brainproducts.com). The activity was recorded at 500 Hz and was referenced online to activity at electrode FCz. Electrolyte gel was used to lower electrode impedances to below 25 kΩ wherever possible. An EOG was recorded using electrodes placed above and below one eye and near the outer canthi of both eyes.

Multiple peripheral physiological measures were collected via a PowerLab 16/35 amplifier and sampled at 1000 Hz by LabChart recording software (ADInstruments, www.adinstruments.com). These included an ECG via three electrodes placed on the shoulders and torso in Lead II configuration; respiration via a respiration belt around the ribcage; EDA through electrode plates placed on the index and ring fingers of the non-dominant hand; and blood volume through an IR plethysmograph attached to the middle finger of the inferior hand. Heart rate (BPM) was calculated online in LabChart using default parameters for human ECG. Hardware filters were applied for the ECG with a low-pass filter of 200 Hz and a high-pass filter of 0.1 Hz, as well as for the plethysmograph with a low-pass filter of 100 Hz; respiration and EDA were not online filtered.



## 1. File naming convention

Each CSV file follows this naming format:

_Features_P###-T##.csv_

Example:

_Features_P040-T21.csv_

This contains data for:

- **Participant ID**: `P040`  
- **Treatment (video stimulus) ID**: `T21`

So:

| Part | Meaning |
|---|---|
P040 | Participant 40 |
T21 | Video stimulus number 21 |

Each file represents **one participant watching one video stimulus**.
There are about ~145 total participants and 24 different treatments.
This seems like a repeated-measure design.

## 2. Video stimulus duration

The videos used in the experiment were short emotional clips.

From the study:

- **Minimum duration:** 13 seconds  
- **Maximum duration:** 19 seconds  
- **Average duration:** ≈15 seconds

However, each video was **repeated four times in a row**.

So the participant actually watched the same stimulus continuously for:

| Video length | Total duration after repetition |
|---|---|
13 s | 52 s |
15 s | 60 s |
19 s | 76 s |

So each trial lasts approximately:

52–76 seconds

## 3. Time binning of the physiological data

The EEG signals were recorded continuously at **500 Hz**, meaning **500 samples were captured per second** for each channel.

To prepare the data for analysis, the continuous recordings were divided into **non-overlapping 500 millisecond (0.5 second) windows**, called **bins**. Because the sampling rate is 500 Hz, each 500 ms bin contains:

500 samples/second × 0.5 seconds = **250 raw samples per channel**

Within each bin, features were computed from these raw samples.

For EEG channels, the **power spectral density (PSD)** was estimated using **Welch’s method**, and the **absolute power** within four frequency bands was calculated as the **area under the PSD curve**:

- Theta: 4–8 Hz  
- Alpha: 8–13 Hz  
- Beta: 13–30 Hz  
- Gamma: 30–60 Hz  

This produces **four features per electrode**. Since the EEG system uses **64 electrodes**, this results in:

64 electrodes × 4 frequency bands = **256 EEG features per bin**

For peripheral physiological signals (heart rate, skin conductance, IR plethysmograph, and respiration), the **mean value within the same 500 ms bin** was computed.

Finally, the participant’s **mean self-reported arousal value during that 500 ms window** was calculated and used as the label for that bin. Each bin therefore becomes **one row in the dataset containing the extracted physiological features and the corresponding arousal value**.

Number of bins per second:

1 second ÷ 0.5 seconds = 2 bins

## 4. Bins per video loop

For a single presentation of the video:

| Video duration | Bins per loop |
|---|---|
13 s | ~26 bins |
15 s | ~30 bins |
19 s | ~38 bins |

## 5. Bins per full trial (4 loops)

Because the clip repeats **four times**, each file contains approximately:

| Video duration | Total bins |
|---|---|
13 s | ~104 rows |
15 s | ~120 rows |
19 s | ~152 rows |

This matches the CSV files where you see roughly **100–150 rows per trial**.

## 6. Structure of one trial

Example with a **15-second clip**:

Loop 1: bins 0–29  
Loop 2: bins 30–59  
Loop 3: bins 60–89  
Loop 4: bins 90–119  

Each row represents **500 ms of recorded physiology**.

## 7. What each row contains

Each row contains:

- **256 EEG features**  
  - EEG band power (Theta, Alpha, Beta, Gamma)
  - computed for **64 electrodes**

- **4 peripheral physiological features**
  - mean heart rate
  - mean skin conductance (GSR)
  - mean IR plethysmograph
  - mean respiration

- **1 label**
  - `LABEL_SR_Arousal`  
  - the participant’s **average reported emotional arousal during that 500 ms window**

Total columns:

256 EEG features  
+ 4 physiological features  
+ 1 arousal label  
= 261 columns

## 8. Why repetition matters

Because the **same video is repeated four times**, the same moment in the video appears multiple times.

For example:

bin 5  
bin 35  
bin 65  
bin 95  

These bins correspond to the **same point in the video**, but during different repetitions.

This means the brain may respond differently each time the stimulus repeats.

Example pattern we may see, assuming **habituation**:

- Loop 1 → strongest emotional response  
- Loop 2 → moderate response  
- Loop 3 → weaker response  
- Loop 4 → stable response  

## 9. Representing the data properly

Instead of treating the trial as a single long sequence, it is better to think of it as:

loops × time_in_video

Example for a 15-second clip:

4 loops × 30 time bins

Each bin contains the EEG features, physiological signals, and arousal rating.

## Key takeaway

Each file represents:

- one participant  
- watching one video stimulus  
- repeated four times  

The dataset records how **brain activity and physiological responses evolve over time** as the same emotional stimulus is repeated.
