# NLP-spam-filter

Learning project in Python for classifying text messages as spam or not spam using a Linear SVC (Support Vector Classifier). The model trains on the SMS Spam Collection Data Set from the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

## Usage
### Packages
The following packages are required. I would strongly recommend using the `pip3` package manager.
* `nltk`
* `pandas`
* `scipy`
* `sklearn`

### Run
To install and run the program, enter the following terminal commands:
```
git clone https://www.github.com/jessicalally/NLP-spam-filter.git
cd NLP-spam-filter
python3 spam_classifier.py
```
The program will then prompt the user to input a message to be classified.

## Examples
```
"Call 01234 567890 NOW for a free pizza"  // This message is spam
"Hi! How are you?"                        // This message is not spam
```
## Analysis

Below are some metrics that display the performance of the model and some of the key terms it identifies in spam messages.

### Confusion Matrix

Out of the 1115 examples used for testing, the model has classified 1094 messages correctly, indicating an accuracy of 0.925. Performance could have been improved by training using a larger dataset, and more comprehensive pre-processing of data.

<table>
<thead>
  <tr>
    <th colspan="2" rowspan="2"></th>
    <th colspan="2">predicted</th>
  </tr>
  <tr>
    <td>spam</td>
    <td>ham</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2"><b>actual</b></td>
    <td>spam</td>
    <td>965</td>
    <td>1</td>
  </tr>
  <tr>
    <td>ham</td>
    <td>20</td>
    <td>129</td>
  </tr>
</tbody>
</table>

### Top Predictors
The following are the 20 top predictors of spam, according to the model.

Each term is a stemmed version of a real word or phrase to generate a more accurate model. A weight is generated by the model for each term which defines how important that term is in determining whether or not a message is spam.

The most influential term is phonenumb (symbolised version of a numerical phone number), meaning the model has identified this as a key factor in detecting spam. Another interesting term is currencysymbolnumbersymbol which symbolises an amount of currency, indicating spam messages will frequently ask or refer to money in their messages.

<table>
<thead>
  <tr>
    <th>Term</th>
    <th>Weight</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>phonenumb</td>
    <td>5.099450</td>
  </tr>
  <tr>
    <td>txt</td>
    <td>2.842654</td>
  </tr>
  <tr>
    <td>call phonenumb</td>
    <td>2.378848</td>
  </tr>
  <tr>
    <td>numbersymbol</td>
    <td>2.220229</td>
  </tr>
  <tr>
    <td>currencysymbolnumbersymbol</td>
    <td>2.005241</td>
  </tr>
  <tr>
    <td>free</td>
    <td>1.821944</td>
  </tr>
  <tr>
    <td>claim</td>
    <td>1.810054</td>
  </tr>
  <tr>
    <td>repli</td>
    <td>1.804426</td>
  </tr>
  <tr>
    <td>rington</td>
    <td>1.801897</td>
  </tr>
  <tr>
    <td>servic</td>
    <td>1.783487</td>
  </tr>
  <tr>
    <td>mobil</td>
    <td>1.733426</td>
  </tr>
  <tr>
    <td>text</td>
    <td>1.620361</td>
  </tr>
  <tr>
    <td>uk</td>
    <td>1.562161</td>
  </tr>
  <tr>
    <td>stop</td>
    <td>1.527382</td>
  </tr>
  <tr>
    <td>tone</td>
    <td>1.494014</td>
  </tr>
  <tr>
    <td>currencysymbolurladdress</td>
    <td>1.436521</td>
  </tr>
  <tr>
    <td>prize</td>
    <td>1.284281</td>
  </tr>
  <tr>
    <td>credit</td>
    <td>1.259734</td>
  </tr>
  <tr>
    <td>order</td>
    <td>1.103612</td>
  </tr>
  <tr>
    <td>poli</td>
    <td>1.087950</td>
  </tr>
</tbody>
</table>

## References
[Using natural language processing to build a spam filter for text messages](https://inmachineswetrust.com/posts/sms-spam-filter/) - Red Huq, In Machines We Trust

[Spam Filtering Emails: An Approach with Natural Language Processing](https://medium.com/@kasumisanchika/spam-filtering-emails-an-approach-with-natural-language-processing-15abb46dd7d5) - Kasumi Gunasekara, Medium @KasumiGunasekara
