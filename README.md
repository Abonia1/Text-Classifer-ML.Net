# Sentiment Analysis ML.NET

Text classification is a smart classification of text into categories. And, using machine learning to automate these tasks, just makes the whole process super-fast and efficient. Artificial Intelligence and Machine learning are arguably the most beneficial technologies to have gained momentum in recent times. 

## Getting Started

It just a model trained with set of 3000 rows of message. Integration phases are there to make it work it out with web applications. For the moment we are storing dataset in. tsv files. Cause that works well with the algorithm we chose to train our model.

### Prerequisites

Visual Studio
Nuget Package:microsoft ML.NET


### TODO 

Because inferred tuple element names are a new feature in C# 7.1 and the default language version of the project is C# 7.0, you need to change the language version to C# 7.1 or higher. To do that, right-click on the project node in Solution Explorer and select Properties. Select the Build tab and select the Advanced button. In the dropdown, select C# 7.1 (or a higher version). Select the OK button.

```
Project ->Properties ->Build->Advanced->Language Version->C# 7.1
```


## Running the tests

For the moment its a console application so you can just run it in console window to get the information about train,evaluate and prediction of the model.

### Train,Test,Deploy

You can train once with your proper dataset only once and it get saved in .Zip format in data folder inside your application.
Test can say about the exact accuracy of the model prediction's result
Deployment is the real prediction phase of given test data


## Deployment

This model can be used as web api inside your application for to predict category of the message

## Built With

* [DatasetKaggle](https://www.kaggle.com/aboniasoja/neocase-case-category-classifier-dataset) - Kaggle dataset for to use with this model V1
* [Asp web applivcation using ML](https://github.com/Abonia1/ML.Net-Model-Integration-with-ASP.NET-Core) - Ongoing POC integrating with asp.net core web application


## Versioning

Minor	Model trained with 30000 messages to detect category
Major	Model trained with large dataset to detect Category and Sub-Category


## Authors

* **Sojasingarayar Abonia** - *Project Initialization*

**Contributors** - *Project Idea* [Fred](https://github.com/fredgodet)




