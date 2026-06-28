% Obliczenie ró¿nicy wartoœci BSS dla podanych kolumn (k¹tów sondowania)
%   dla poszczególnych sondowañ
pierwsza_dana = 1;
liczba_danych = 500;
kat_max = 80;  % nr kolumny; Odpowiada to w przybli¿eniu k¹towi wi¹zki 0 stopni.
kat_min = 100;  % nr kolumny; Odpowiada to w przybli¿eniu k¹towi wi¹zki 20 stopni.
load dane_bss
%   £aduje siê zmienna vbss - tablica 3-wymiarowa o rozmiarach: liczba
%     typów dna (= 4), liczba sondowañ (danych) (= 600), liczba wi¹zek
%     (wartoœci bss) w 1 sondowaniu (= 160).
[ltypowdna lswathow lbeamow] = size(vbss);
% obliczenie wartoœci ró¿nicy BSS dla kat_min i kat_max dla poszczególnych swath'ów
clear var_spadek_dB
for itypdna = 1 : ltypowdna
  for idana = pierwsza_dana : pierwsza_dana - 1 + liczba_danych  % Zak³adamy, ¿e nie przekroczymy tu lswathow.
    var_spadek_dB(itypdna, idana) = vbss(itypdna, idana, kat_max) - vbss(itypdna, idana, kat_min);
  end  % for idana
end  % for itypdna
