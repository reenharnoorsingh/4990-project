# 4990 Project

# Prerequisites

1. python/python3

2. pip/pip3

3. django


# Guide to Install and Run program

$ git clone https://github.com/reenharnoorsingh/4990-project.git

$ cd 4990-Project

# Creating virtual environment
To install virtual environment you can use pip (pip3 install virtualenv)

1. $ virtualenv env
1. $ python3 -m venv env ( For Windows )

2.  source env/bin/activate
2.  .\env\Scripts\activate ( For Windows )

# Downloading all the requirements

$ pip3 install -r requirements.txt

# Setting up database

$ python3 manage.py makemigrations (when you change the database)

$ python3 manage.py migrate

# Create the Superuser

$ python3 manage.py createsuperuser

# Start the app

$ python3 manage.py runserver

## Codebase structure


```
< PROJECT ROOT >
   |
   |-- core/                            
   |    |-- settings.py    # Project Configuration  
   |    |-- urls.py        # Project Routing
   |
   |-- home/
   |    |-- views.py       # APP Views
   |    |-- urls.py        # APP Routing
   |    |-- models.py      # APP Models
   |    |-- tests.py       # Tests  
   |
   |-- requirements.txt    # Project Dependencies
   |
   |-- env.sample          # ENV Configuration (default values)
   |-- manage.py           # Start the app - Django default start script
   |
   |-- ************************************************************************


< UI_LIBRARY_ROOT >                      
   |
   |-- templates/                     # Root Templates Folder
   |    |          
   |    |-- accounts/       
   |    |    |-- auth-signin.html     # Sign IN Page
   |    |    |-- auth-signup.html     # Sign UP Page
   |    |
   |    |-- includes/       
   |    |    |-- footer.html          # Footer component
   |    |    |-- sidebar.html         # Sidebar component
   |    |    |-- navigation.html      # Navigation Bar
   |    |    |-- scripts.html         # Scripts Component
   |    |
   |    |-- layouts/       
   |    |    |-- base.html            # Masterpage
   |    |
   |    |-- pages/       
   |         |-- dashboard.html       # Dashboard page
   |         |-- user.html            # Settings  Page
   |         |-- *.html               # All other pages
   |    
   |-- ************************************************************************
