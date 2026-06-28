% Obliczenie "six point height" g³êbokoœci dla danego sondowania dla podanego zakresu kolumn (k¹tów)
% Jest to ró¿nica pomiêdzy œredni¹ z 3 najwy¿szych punktów i œredni¹ z 3
%   najni¿szych punktów.
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
% obliczenie "sixph" g³êbokoœci dla poszczególnych swath'ów
clear var_sixph
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
    end    % obliczenie "sixph" dla danego typu dna i swath'u
    z_sorted = sort(fragm_obrazu_do_stat);
    z_min_ = mean(z_sorted(1 : 3));
    z_max_ = mean(z_sorted(z_l - 2 : z_l));
    var_sixph(itypdna, idana) = z_max_ - z_min_;
  end  % for idana
end  % for itypdna
