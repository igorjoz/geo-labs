% Klasyfikacja minimalnoodleg³oœciowa na podstawie obliczonych wartoœci 2 parametrów
% Korzysta z odleukkw.m.
zm1 = var_srednia;  % pierwszy parametr (--> oœ x)
%zm2 = var_spadek_dB;  % drugi parametr (--> oœ y)
%zm1 = var_GLCM_entr;  % pierwszy parametr (--> oœ x)
zm2 = var_GLCM_spoj;  % drugi parametr (--> oœ y)
% Rozmiar zm1 i zm2 musi byæ taki sam.
[ltypowdna, lobiektow] = size(zm1);  % ltypowdna - liczba klas
                                     % lobiektow - liczba obiektów w 1 klasie (taka sama w ka¿dej)
wspzbucz = 20;  % [%]; Ile % danych ma stanowic zbiór ucz¹cy?

% normalizacja
zmien1 = (zm1 - min(min(zm1))) / (max(max(zm1)) - min(min(zm1)));
zmien2 = (zm2 - min(min(zm2))) / (max(max(zm2)) - min(min(zm2)));

% okreslenie zbioru ucz¹cego i testowego
% pierwsze lobucz obiektów - zbiór ucz¹cy, reszta - zbiór testowy
lobucz = lobiektow * wspzbucz / 100;  % liczba obiektów w zbiorze ucz¹cym w 1 klasie
zm1_ucz = zmien1(:, 1 : lobucz);
zm2_ucz = zmien2(:, 1 : lobucz);
zm1_test = zmien1(:, lobucz+1 : lobiektow);
zm2_test = zmien2(:, lobucz+1 : lobiektow);
[ltypowdna, lobtest] = size(zm1_test);  % lobtest - liczba obiektów w zbiorze testowym w 1 klasie
% Jeœli wspzbucz = 20 i lobiektow = 500, to lobucz bêdzie równe 100, a
% lobtest bêdzie równe 400.

% obliczenie œrodków skupieñ dla zbioru ucz¹cego
% Rozmiar zmiennych srodki_zmx bêdzie: ltypowdna na 1 czyli 4 na 1.
srodki_zm1 = mean(zm1_ucz, 2);  % dla zmiennej 1
srodki_zm2 = mean(zm2_ucz, 2);  % dla zmiennej 2

% Klasyfikacja
% obliczenie (kwadratu) odleg³oœci euklidesowej od ka¿dego obiektu w zbiorze
%   testowym do ka¿dego œrodka klasy ze zbioru ucz¹cego, w przestrzeni
%   parametrów (zmien1, zmien2), i zapisanie do zmiennej odltestucz
% Rozmiar zmiennej odltestucz bêdzie: ltypowdna na lobtest na ltypowdna 
% czyli 4 na lobtest na 4.
clear odltestucz
for itypdna = 1 : ltypowdna
  for iobtest = 1 : lobtest  
    for itypdnaucz = 1 : ltypowdna  
      odltestucz(itypdna, iobtest, itypdnaucz) = odleukkw([zm1_test(itypdna, iobtest) zm2_test(itypdna, iobtest)], ...
        [srodki_zm1(itypdnaucz) srodki_zm2(itypdnaucz)]);
    end
  end
end

% wybór klasy wynikowej dla poszczeg. elementów ze zbiotu testowego i zapisanie jej w klaswyn
% Rozmiar zmiennej klaswyn bêdzie: ltypowdna na lobtest czyli 4 na lobtest.
clear klaswyn
for itypdna = 1 : ltypowdna
  for iobtest = 1 : lobtest
    [minodl klasa] = min(odltestucz(itypdna, iobtest, :));
    klaswyn(itypdna, iobtest) = klasa;
  end  
end

clear macierzniezgodnosci
% obliczenie macierzy niezgodnoœci
% Rozmiar zmiennych macierzniezgodnosci i macierzniezgodnosciproc bêdzie: ltypowdna na ltypowdna 
%   czyli 4 na 4.
for itypdna = 1 : ltypowdna
  for itypdnaklas = 1 : ltypowdna  
    macierzniezgodnosci(itypdna, itypdnaklas) = length(find(klaswyn(itypdna, :) == itypdnaklas));
  end
end
macierzniezgodnosciproc = macierzniezgodnosci / lobtest * 100; % [%]
procpoprawnych = sum(diag(macierzniezgodnosci)) / (ltypowdna * lobtest) * 100;

% wydruk wyników
disp(macierzniezgodnosci)
disp(macierzniezgodnosciproc)
disp('Razem poprawnych klasyfikacji [%]:')
disp(procpoprawnych)
