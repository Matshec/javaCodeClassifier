#Java code classifier
This is java code classifier that is able to clasify java lines of code as general programming concepts
like classes, loops, functions etc


For parser to create appropriate datafile look [to this repo here](https://github.com/falanadamian/java-parser) 

### Prequisitions
python 3.6 and following libraries
* keras 2.*
* tensorflow 2.*
* numpy
* scikit-learn
* pandas

## Run
`python3 main.py <path to .csv parsed file>`

##TODO
tweaks, add context to classification

### Unofficial
polish only

Na ta chwile model jest bardzo prosty ale dosyc skuteczny - ilosc warstw  wzieta z powietrza

processing danych:

-> wczytanie danych 
 
-> obciecie lini z kodem do max 100 znakow ( wiece raczej nie jest konieczne na teraz - moze z kontekstem) 

-> encodeowanie danych(lini z kodem):
* podzielenie calego stringu w miejscach spacji i kropek (split)
* mapowanie slow kluczowych na wartosci liczbowe (int)
 przy czym wszystkie slowa nie bedea kluczowymi sa mapowane na jedna wartosc - 1 a srednik jest uznany za slowo kluczowe
* sprowadzenie listy z pkt wyzej do takiego wektora ze pozycja odpowiada za typ slowa, a wartosc to ilosc powtorzen tego
slowa (opcja do moze zmiany)
 
 -> encodowanie labelek - one-hot encoding
 
 -> uczenie modelu z wykorzystaniem crossvalidation
