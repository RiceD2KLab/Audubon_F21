### Requirements to run the project

To run this project on your local machine you need 4 things
1. This project must be on your computer
2. Node.js and npm must be installed on your computer
3. Python3 and several dependencies must be installed on your computer
4. The 2 models must be downloaded and installed a folder within this project

### Installing Node.js and npm

In order to install Node.js and npm you would need to go to https://nodejs.org/en/download and pick the package that is correct for your os. If you are using a windows computer click on Windows Installer. And if you are using a Mac click on macOS installer. Follow the prompts, no need for any change in the settings. Once you are done with the installer both Node.js and npm should be installed.

### Python3 and pip

In order to install python3 go to this website https://www.python.org/downloads/ and follow the installation instructions similar to when installing Node.js. Once you are done with the installer both python3 and pip should be installed on your computer

### Installing the Detection and Classification models

Within this project go to the folder titled GUI/server. If no models folder is found there make one. Then go to this link, https://drive.google.com/drive/folders/1GkywYLXdK-BeN7aoqSK5UsFyJ24X6Tcn?usp=sharing, and download both of the models to the models folder that was stated above.

### Installing all dependencies

There are two sets of dependencies to install. First, to install the react dependencies in computer terminal navigate to the 'audobon-website' directory. Once there type in the command 'npm install'. This command will look at the package.json file, find all the dependencies needed and install them automatically. It will take a while.

Next to the install the python dependencies, in your terminal type in this series of commands:
pip install flask
pip install flask_restful
pip install flask-cors
pip install pandas
pip install ast
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117   
pip install opencv-python 

Upon following all of the above steps the program should be good to run.

### Running the project
In order to run the project you need two terminals, one for the webpage and one for the backend server. 
To run the webpage go back to the 'audobon-website' folder in your terminal and type in the command 'npm start'. This will take a while to run the first time as react sets modules up. Once it is done a window should open in your default browser to the url 'http://localhost:3000/', displaying the landing page.

To run the backend server you can right click on the 'script.py' file and copy its path (important to not copy the relative path) and paste that path into a different terminal. Alternatively you can hit run on an IDE while you have that file open. After a couple of seconds you should see:
        * Serving Flask app 'restapi'
        * Debug mode: off
        WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
        * Running on http://127.0.0.1:5000
        Press CTRL+C to quit
on your terminal, this means that it is running successfully.

Congrats! You have now set up the application and can begin testing.