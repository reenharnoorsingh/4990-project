4990 Project

Prerequisites

python/python3
pip/pip3
django


Guide to Install and Run program

1. $ git clone https://github.com/reenharnoorsingh/4990-project.git

2. $ cd 4990_Project

# Creating virtual environment
To install virtual environment you can use pip (pip3 install virtualenv)

3. $ virtualenv env
3. $ python3 -m venv env ( For Windows )

4. $ source env/bin/activate
4. $ .\env\Scripts\activate ( For Windows )

# Downloading all the requirements

5. $ pip3 install -r requirements.txt

# setting up database

6. $ python3 manage.py makemigrations (when you change the database)

7. $ python3 manage.py migrate

# Create the Superuser

8. $ python3 manage.py createsuperuser

# Start the app

$ python3 manage.py runserver

## Codebase structure

The project is coded using a simple and intuitive structure presented below:

```bash
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
