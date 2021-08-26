#Experiments
##1

###Config
>{ 
    model: bart-base,
    dataset: cnn,
    dataset_part_for_training: full,
    epochs_num: 10,
    input: arange_input
}

###Metrics
>metrics: {
    rouge1: 26.976, 
    rouge2: 10.8858, 
    rougeL: 20.4431
}

###Some examples of summaries

>Generated
> >James Best, 88, died Monday.
> 
>Label
> >James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88.
"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV.

>Generated
> >TePCO says the site's still too dangerous for workers to enter.
> 
>Label
> >A robotic probe into the Fukushima nuclear plant released crucial information on conditions inside the reactor.
TEPCO: Recorded radiation levels and temperatures are lower than expected.
The robot was sent into the plant after the first one broke down.

>Generated
> >President Obama: "No challenge poses more of a public threat than climate change"
> 
>Label
> >"No challenge poses more of a public threat than climate change," the President says.
He credits the Clean Air Act with making Americans "a lot" healthier.

##2

###Config
>{ 
    model: bart-base,
    dataset: cnn,
    dataset_part_for_training: 1/100,
    epochs_num: 10,
    input: arange_input
}

###Metrics
>metrics: {
    'rouge1': 23.1791, 
    'rouge2': 9.1714, 
    'rougeL': 17.3658
}

###Some examples of summaries

>Generated
> >Actor Jimmy Best was best known for his "hot pursuit"
> 
>Label
> >James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88.
"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV.

>Generated
> >TePCO says the site's still too dangerous for workers to enter.
> 
>Label
> >A robotic probe into the Fukushima nuclear plant released crucial information on conditions inside the reactor.
TEPCO: Recorded radiation levels and temperatures are lower than expected.
The robot was sent into the plant after the first one broke down.

>Generated
> >President Obama says the Clean Air Act and subsequent amendments have reduced early deaths.
> 
>Label
> >"No challenge poses more of a public threat than climate change," the President says.
He credits the Clean Air Act with making Americans "a lot" healthier.

##3

###Config
>{ 
    model: bart-large,
    dataset: cnn,
    dataset_part_for_training: 1/100,
    epochs_num: 10,
    input: arange_input
}

###Metrics
>metrics: {
    'rouge1': 36.7568, 
    'rouge2': 15.6933, 
    'rougeL': 25.9815
}

###Some examples of summaries

>Generated
> >James Best, 88, died Monday after a brief illness.
He played sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard"
>
>Label
> >James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88.
"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV.

>Generated
> >Novak Djokovic beats Thomas Berdych 7-5, 4-6, 6-3 in the Monte Carlo Masters final.
Djokovic is the first man to win the opening three Masters tournaments of the season.
> 
>Label
> >Djokovic wins Monte Carlo Masters.
Defeats Berdych 7-5, 4-6, 6-3.
Djokovic had earlier beaten clay expert Nadal in semis.

>Generated
> >Andrew Chan married Febyanti Herewila, his girlfriend of three years.
The couple met in Kerobokan prison in 2012 after a friend introduced the pair.
Myuran Sukumaran has painted a series of haunting self-portraits.
> 
>Label
> >Bali Nine ringleader Andrew Chan has married fiance Febyanti Herewila.
The pair wed at Besi Prison on Nusakambangan Island on Monday.
Chan proposed to Febyanti in February while he was still at Kerobokan.
He and Myuran Sukumaran are set to be executed on Wednesday morning.

##4

###Config
>{ 
    model: bart-large with extra tanh and linear,
    dataset: cnn,
    dataset_part_for_training: 1/100,
    epochs_num: 10,
    input: arange_input
}

###Metrics
>metrics: {
    'rouge1': 36.2712, 
    'rouge2': 15.4233, 
    'rougeL': 25.6864
}

###Some examples of summaries

>Generated
> >James Best, 88, died Monday after a brief illness.
Best played sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard"
>
>Label
> >James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88.
"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV.

>Generated
> >Novak Djokovic wins Monte Carlo Masters.
Djokovic beats Thomas Berdych 7-5, 4-6, 6-3 in a rain-interrupted final.
> 
>Label
> >Djokovic wins Monte Carlo Masters.
Defeats Berdych 7-5, 4-6, 6-3.
Djokovic had earlier beaten clay expert Nadal in semis.

>Generated
> >Andrew Chan and Myuran Sukumaran are set to be executed at 3am AEST on Wednesday.
Febyanti and Chan met each other in Kerobokan prison in 2012 after a friend introduced them.
> 
>Label
> >Bali Nine ringleader Andrew Chan has married fiance Febyanti Herewila.
The pair wed at Besi Prison on Nusakambangan Island on Monday.
Chan proposed to Febyanti in February while he was still at Kerobokan.
He and Myuran Sukumaran are set to be executed on Wednesday morning.

##5

###Config
>{ 
    model: bart-large,
    dataset: cnn,
    dataset_part_for_training: 1/100,
    epochs_num: 10,
    input: random_input
}

###Metrics
>metrics: {
    'rouge1': 37.0196, 
    'rouge2': 16.0939, 
    'rougeL': 26.2081
}

###Some examples of summaries

>Generated
> >James Best died in hospice in Hickory, North Carolina, of complications from pneumonia.
Best played sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard"
>
>Label
> >James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88.
"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV.

>Generated
> >Novak Djokovic beats Thomas Berdych 7-5, 4-6, 6-3 in rain-interrupted final of Monte Carlo Masters.
Djokovic extends his winning streak to 17 matches.
Berdych's third loss in a final this year.
> 
>Label
> >Djokovic wins Monte Carlo Masters.
Defeats Berdych 7-5, 4-6, 6-3.
Djokovic had earlier beaten clay expert Nadal in semis.

>Generated
> >Andrew Chan married Febyanti Herewila, his girlfriend of three years.
The pair met in Kerobokan prison in 2012 after a friend introduced the pair.
Myuran Sukumaran is set to be executed on Wednesday.
> 
>Label
> >Bali Nine ringleader Andrew Chan has married fiance Febyanti Herewila.
The pair wed at Besi Prison on Nusakambangan Island on Monday.
Chan proposed to Febyanti in February while he was still at Kerobokan.
He and Myuran Sukumaran are set to be executed on Wednesday morning.

##6

###Config
>{ 
    model: bart-large,
    dataset: cnn,
    dataset_part_for_training: 1/50,
    epochs_num: 7,
    input: random_input
}

###Metrics
>metrics: {
    'rouge1': 37.4084, 
    'rouge2': 16.3604, 
    'rougeL': 26.5555
}

###Some examples of summaries

>Generated
> >James Best, 88, died Monday after a brief illness.
He played sheriff Rosco P. Coltrane on TV's "The Dukes of Hazzard"
Best's character became known for his distinctive "kew-kew" chuckle and for goofy catchphrases.
>
>Label
> >James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88.
"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV.

>Generated
> >Novak Djokovic beats Thomas Berdych 7-5, 4-6, 6-3 in the Monte Carlo Masters final.
Djokovic is the first man to win the opening three Masters tournaments of the season.
> 
>Label
> >Djokovic wins Monte Carlo Masters.
Defeats Berdych 7-5, 4-6, 6-3.
Djokovic had earlier beaten clay expert Nadal in semis.

>Generated
> >Andrew Chan has married his fiancee Febyanti Herewila.
The pair wed inside the chapel at Besi Prison on Nusakambangan Island on Monday.
Myuran Sukumaran is set to be executed at 3am AEST on Wednesday.
> 
>Label
> >Bali Nine ringleader Andrew Chan has married fiance Febyanti Herewila.
The pair wed at Besi Prison on Nusakambangan Island on Monday.
Chan proposed to Febyanti in February while he was still at Kerobokan.
He and Myuran Sukumaran are set to be executed on Wednesday morning.

##7

###Config
>{ 
    model: bart-large,
    dataset: xsum,
    dataset_part_for_training: 1/100,
    epochs_num: 10,
    input: random_input
}

###Metrics
>metrics: {
    'rouge1': 36.7893, 
    'rouge2': 15.2479, 
    'rougeL': 29.7982
}

###Some examples of summaries

>Generated
> >A new community hospital in Somerset is to be built.
>
>Label
> >A community hospital in Somerset is to be replaced and rebuilt with a £16m grant from the government.

>Generated
> >A leading Australian politician has proposed a four-day week.
> 
>Label
> >Australia should consider adopting a four-day working week or a six-hour working day, a political leader says.

>Generated
> >A Brazilian restaurant worker has said he was "sad" after he was deported for using false documents to get a job at Byron restaurants.
> 
>Label
> >A former worker at the Byron hamburger chain, who was arrested and deported after immigration raids last month, says he feels "used".