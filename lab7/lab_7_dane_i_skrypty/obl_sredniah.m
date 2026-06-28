% Obliczenie œredniej g³êbokoœci dla danego sondowania dla podanego zakresu kolumn (k¹tów)
pierwsza_dana = 1;
liczba_danych = 500;
kolumny = 41 : 80;  % Dla jakiego zakresu kolumn-wi¹zek liczymy?
detrending = 1;  % Czy usuwamy trend - odemujemy aproksymowany liniowy przebieg 
%   dna (sk³adow¹ liniowo narastaj¹c¹) dla ka¿dego sondowania?
%detrending = 0;
load dane_z
% £aduje siê vz - tablica 3D zawieraj¹ca dane o g³êbokoœci 
%   dna zmierzonej sonarem wielowi¹zkowym, o rozmiarach: liczba
%   typów dna (= 4), liczba sondowañ (danych) (= 600), liczba wi¹zek
%   w 1 sondowaniu (= 160).
[ltypowdna lswathow lbeamow] = size(vz);
clear var_sredniah
for itypdna = 1 : ltypowdna
  for idana = pierwsza_dana : pierwsza_dana - 1 + liczba_danych  % Zak³adamy, ¿e nie przekroczymy tu lswathow.
    % wybranie fragmentu obrazu (fragment 1 poziomej linii 1 obrazu-typu dna, 
    %   dla danego zakresu kolumn-wi¹zek)
    fragm_obrazu_do_stat = reshape(vz(itypdna, idana, kolumny), 1, length(kolumny));
    z_l = length(fragm_obrazu_do_stat);
    % detrending
    if detrending
      a = polyfit(1 : z_l, fragm_obrazu_do_stat, 1);
      fragm_obrazu_do_stat = fragm_obrazu_do_stat - a(1) * (1:z_l) - a(2);
    end
    % obliczenie sredniej g³êbokoœci dla danego typu dna i swath'u
    var_sredniah(itypdna, idana) = mean(fragm_obrazu_do_stat);
  end  % for idana
end  % for itypdna
