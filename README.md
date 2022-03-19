# template_with_mlflow
This is a template repository for building a model and managing the trials with google colaboratory, google drive, and mlflow. We could see how amazing performing experiments with mlflow is.

## Quickstart
I explained how to use this repository in 
[[Using MLflow on google colaboratory with github to build cosy environment: building](https://dev.to/ksk0629/using-mlflow-on-google-colaboratory-with-github-to-build-cosy-environment-building-jb5)] too. Most of the parts are the same as here, but the article has some pictures being to understand easier. If you are interested in it, please see it. I would appreciate it if you could leave comments.

### Preparation
1. Creating accounts

I'm quite sure someone who is seeing this repository has already got the account on git. Except for the account, we have got to create accounts on google and ngrok to start this quickstart.

2. Getting a personal access token and an authentication token

The first one is from git. We could see how to create it on the official page [[Creating a personal access token]](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). Another token is found on a ngrok top page, like the following line.
```
$./grok authtoken [YOUR_TOKEN]
```

3. Uploading a config file to a `config` folder on a google drive

First, we have to create a `config` folder on google drive and then create and upload a config file. The config file should be named `general_config.ymal` and constructed like the following lines.

```yaml
github:
  username: your_username
  email: your_email@gmail.com
  token: your_personal_access_token
ngrok:
  token: ngrok_authentication_token
```

### Performing experiment
1. Cloning this repository on a local machine

2. Uploading `template_with_mlflow.ipynb` to a google drive

3. Running all cells

After the cell in Run MLflow section was run, we could see the results of experiments on the outputted URL, like `MLflow Tracking UI: https://xxx-xx-xxx-xxx-xx.ngrok.io.`.