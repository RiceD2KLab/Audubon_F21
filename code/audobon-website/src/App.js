import logo from './TexasAudubonLogo.png';
import './App.css';
import React, {useState} from 'react';
import axios from 'axios';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';

/***
 * The main webpage. Contains all of the data with regards to the webpage and sets up communication to the backend
 */
function App() {
  
  //List of the array with all of the bird species to be selected from in the display
  const birdOptions = [
    'American Avocet Adult AMAVA',
    'Brown Pelican Wings Spread AMOYA',
    'Black-Crowned Night Heron Adult BCNHA',
    'Black Skimmer Adult BLSKA',
    'Brown Pelican Adult BRPEA',
    'Brown Pelican Chick BRPEC',
    'Brown Pelican Juvenile BRPEJ',
    'Brown Pelican - Wings Spread BRPEW',
    'Cattle Egret Adult CAEGA',
    'Double-Crested Cormorant Adult DCCOA',
    'Great Blue Heron Adult GBHEA',
    'Great Blue Heron Chick GBHEC',
    'Great Blue Heron Egg GBHEE',
    'Great Blue Heron Juvenile GBHEJ',
    'Black Crowned Night Heron Adult GBHEN',
    'Great Egret Adult GREGA',
    'Great Egret Chick GREGC',
    'American Oystercatcher GREGF',
    'Laughing Gull Adult LAGUA',
    'Laughing Gull Juvenile LAGUJ',
    'Great Egret/White Morph Adult MEGRT',
    'Mixed Tern Adult MTRNA',
    'Other Bird OTHRA',
    'Reddish Egret Adult REEGA',
    'Reddish Egret Chick REEGF',
    'White Morph Reddish Egret Adult REEGWMA',
    'Roseate Spoonbill Adult ROSPA',
    'Snowy Egret SNEGA',
    'Trash/Debris TRASH',
    'Tricolored Heron Adult TRHEA',
    'White Ibis Adult WHIBA',
    'White Ibis Chick WHIBC',
    'White Ibis Juvenile WHIBJ',
  ]

  // Set up all variables to be used
  const [file, setFile] = useState(null)
  const [numbirds, setNumBirds] = useState(0)
  const [success, setSuccess] = useState(false)
  const [selectedBird, setSelectedBird] = useState(birdOptions[0])
  const zoomed_in_image = require.context('../../server/upload/', true)
  const [birdNum, setBirdNum] = useState(0)
  const [nameArray, setNameArray] = useState([''])
  const [inputVal, setACInputVal] = useState('')
  const [csvData, setCSVData] = useState([['', '', 0, 0, 0, 0], ['AMAVA', 'American Avocet Adult', 0, 0, 0, 0]])
  
  /**
   * Stores the uplaoded file
   * Inputs:
   *      e - An object that contains the data for the file that was uploaded by the user
   * Returns:
   *      An updated file variable containing the file that was uploaded
   */
  const changedFile = (e) => {
    setFile(e.target.files[0]);
  };

  /**
   * Sends the uploaded file to the backend to be run through the detector and classifier and sets up
   * the rest of the GUI for annotation
   * Inputs:
   *      file - An object containing the image that was uploaded by the user
   * Returns:
   *      The data from the classifier in the form of csvData, nameArray, numBirds
   */
  const uploadedFile = () => {
    if (file == null) {
      return;
    }

    const imgUpload = new FormData();

    imgUpload.append("newIMG", file);

    console.log(file.name);
    
    axios.post('http://127.0.0.1:5000/images', imgUpload, {
      headers: {
        'Content-Type': imgUpload.getHeaders
      },
    }).then((response) => {
      setNumBirds(response.data.num_birds)
      setNameArray(response.data.bird_names)
      setCSVData(response.data.data)
      setSuccess(true)
      setSelectedBird(birdOptions.find(element => element === response.data.bird_names[0]))
    }).catch((error) => {
      if (error.response){
        console.log("error:");
        console.log(error.response.data);
      }
    });
  };
  
  /**
   * Changes the selected bird in accordance with user input
   * Inputs:
   *      e - An object containing information found in the combobox that the user is changing
   * Returns:
   *      New value for selectedBird ina ccordance to what the user changed
   */
  const changeBird = (e) => {
    let bird = e.target.innerHTML;
    setSelectedBird(bird)
  }

  /**
   * Advances to the next bird and should mark the current image as not a bird
   * Input:
   *     birdNum - An integer value representing the bird we are currently working on
   * Returns:
   *     An updated csvData array denoting the current bird as trash
   */
  const notABird = () => {
    console.log("Not a bird!")
    csvData[birdNum + 1][0] = "TRASH"
    csvData[birdNum + 1][1] = "Trash/Debris"
    
    nextBirdFunc()
  }

  /**
   * Updates the csvDataArray to display the new data that was presented to it
   * Inputs:
   *      selectedBird - A string denoting the value selected by the user
   *      csvData - An array containing the information that is being stored by the system
   *      birdNum - An integer value representing the bird that we are working on
   * Returns:
   *      An updated csvData array in accordance to user input
   */
  const sendInfo = () => {
    let birdArr = selectedBird.split(' ')

    csvData[birdNum + 1][0] = birdArr.pop()
    csvData[birdNum + 1][1] = birdArr.join(' ')

    nextBirdFunc()
  }

  /**
   * Advances to the next bird, setting all variables up to display it
   * Inputs:
   *      birdNum - An integer value representing the bird that we are on
   * Returns:
   *      A new screen displaying the next bird in the list
   */
  const nextBirdFunc = () => {
    let x = birdNum + 1
    if (x === numbirds) {
      return
    }
    setBirdNum(x)
    setSelectedBird(birdOptions.find(element => element === nameArray[x]))
  }

  /**
   * Goes back to the next bird upon button press, setting all variables up to display it
   * Inputs:
   *      birdNum - An integer value representing the bird that we are on
   * Returns:
   *      A new screen displaying the previous bird in the list
   */
  const prevBirdFunc = () => {
    let x = birdNum - 1
    if (x < 0) {
      return
    }
    setBirdNum(x)
    setSelectedBird(birdOptions.find(element => element === nameArray[x]))
  }

  /**
   * Resets everything in the page to prepare for a new file to be uploaded
   */
  const doneWithImage = () => {
    setSuccess(false)
    setBirdNum(0)
    setNameArray([])
    setSelectedBird('')
    setNumBirds(0)
    axios.post('http://127.0.0.1:5000/delete', {
      birdNum: numbirds,
      fileName: file.name,
    })
  }

  /**
   * Generates a CSV to be downlaoded for the user upon button click
   * Inputs:
   *      csvData - An array which contains all of the data collected from this image
   * Returns:
   *      A CSV to be downloaded
   */
  const getCSV = () => {
    console.log("Hello")
    let csvContent = "data:text/csv;charset=utf-8," 
      + csvData.map(e => e.join(",")).join("\n");
    var encodedUri = encodeURI(csvContent)
    window.open(encodedUri)
  }

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Audubon Texas's Annotation Assistant
        </p>
        <div style={{display: success ? 'none' : 'inline'}}>
          <form id = 'fileUpload' action="/profile" method="post" encType="multipart/form-data">
            <input type="file" onChange={changedFile} />
          </form>
          <button onClick={uploadedFile}>
            File Upload
          </button>
        </div>
        <div className="Model-Output" style={{display: success ? 'inline' : 'none'}}>

          <div className="Bird-Class">
            AI Prediction: {nameArray[Number(birdNum)]}
            <div className="Images">
              <img src={zoomed_in_image(`./bird${birdNum}.jpg`)} className="zoomed-img" alt="bird"/>
              <img src={zoomed_in_image(`./expanded_bird${birdNum}.jpg`)} className="surroundings" alt="surroundings" />
            </div>
            <div className="Questions">
              <label>
                What bird is this?
              </label>
              <Autocomplete
                id = "bird-options"
                options = {birdOptions}
                onChange={changeBird}
                value = {[csvData[birdNum + 1][1], csvData[birdNum + 1][0]].join(' ')}
                inputValue = {inputVal}
                onInputChange={(_, newInputValue) => {
                  setACInputVal(newInputValue)
                }}
                renderInput={(params) => <TextField {...params}
                  if = "bird-textfield"
                  label="Birds" 
                  color='warning'/>}
              />
              <button onClick={sendInfo}>
                Send
              </button>
              <button onClick={notABird}>
                Not A Bird
              </button>
            </div>
            <button onClick = {prevBirdFunc}>
              Previous Bird
            </button>
            <button onClick = {nextBirdFunc}>
              Next Bird
            </button>
            <button onClick = {getCSV}>
              Get CSV
            </button>
            <button onClick = {doneWithImage}>
              Done With File
            </button>
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
