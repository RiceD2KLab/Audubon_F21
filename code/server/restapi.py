from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import os
import sqlconnection as sql
from flask_cors import CORS
from werkzeug.utils import secure_filename
import script

UPLOAD_FOLDER = r'C:\ProgramData\MySQL\MySQL Server 8.0\Uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

api = Api(app)
CORS(app)
@app.route("/")

class Images(Resource):
    
    '''
    Causes the image to be run through the model and returns the data that is returned by said model
    '''
    def post(self):
        files = request.files
        image = files.get('newIMG')

        filename = secure_filename(image.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("Path saved: " + filename)
        image.save(path)

        nameArray = script.bird_classifier(path)

        for data in nameArray:
            print(data)

        return {'data': len(nameArray),
                'bird_names': nameArray}, 200
     
class Annotations(Resource):
    '''
    Updates the data found in the database in accordance to the new classification inputted by the user
    '''
    def post(self):
        data = request.json
        sql.updateRow(data["num"], data["birdCode"])
        return {'data': "Hi"}, 200
    
    '''
    Calls to the database to generate a csv file to be sent to the frontend
    '''
    def get(self):
        sql.getCSV()
        return {'data': "Hello"}, 200
    
class Delete(Resource):

    '''
    Deletes all images generated for an image that is now done being used
    '''
    def post(self):
        data = request.json
        num = data['birdNum']

        for i in range(1, num):
            os.remove(r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\upload\expanded_bird' + str(i) + ".jpg")
            os.remove(r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\upload\bird' + str(i) + ".jpg")
            

        print(data)


api.add_resource(Annotations, '/annotations')
api.add_resource(Images, '/images')
api.add_resource(Delete, '/delete')

if __name__ == '__main__':
    app.run()  # run our Flask app