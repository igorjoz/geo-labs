% Obliczenie œredniej wartoœci BSS dla podanego zakresu kolumn (k¹tów)
pierwsza_dana = 1;
liczba_danych = 500;
kolumny = 80 : 100;  % Odpowiada to w przybli¿eniu k¹tom wi¹zek od 0 do 20 stopni.
load dane_bss
%   £aduje siê zmienna vbss - tablica 3-wymiarowa o rozmiarach: liczba
%     typów dna (= 4), liczba sondowañ (danych) (= 600), liczba wi¹zek
%     (wartoœci bss) w 1 sondowaniu (= 160).
[ltypowdna lswathow lbeamow] = size(vbss);
% obliczenie œrednich bss dla swath'ów
clear var_srednia
for itypdna = 1 : ltypowdna
  for idana = pierwsza_dana : pierwsza_dana - 1 + liczba_danych  % Zak³adamy, ¿e nie przekroczymy tu lswathow.
    % wybranie fragmentu obrazu (fragment 1 poziomej linii 1 obrazu-typu dna, 
    %   dla danego zakresu kolumn-wi¹zek)
    fragm_obrazu_do_stat = vbss(itypdna, idana, kolumny);
    % obliczenie œredniej dla danego typu dna i swath'u
    var_srednia(itypdna, idana) = mean(fragm_obrazu_do_stat);
  end  % for idana
end  % for itypdna
