% Wyœwietlenie "seabed (xy)z data" - siatki powierzchni dna morskiego
z_zakres = [-0.7 0.7];
%detrending = 0;  % Czy usuwamy trend - odemujemy aproksymowany liniowy przebieg 
%   dna (sk³adow¹ liniowo narastaj¹c¹) dla ka¿dego sondowania?
detrending = 1;
swathsektor = 51:80;  % przyk³adowy fragment powierzchni dna
beamsektor = 41:80;
load dane_z  % Wczytuje siê zmienna vz - tablica 3D zawieraj¹ca dane o g³êbokoœci 
%   dna zmierzonej sonarem wielowi¹zkowym, o rozmiarach: liczba
%   typów dna (= 4), liczba sondowañ (danych) (= 600), liczba wi¹zek
%   w 1 sondowaniu (= 160).
[ltypowdna lswathow lbeamow] = size(vz);
% wyœwietlenie obrazów
for itypdna = 1 : ltypowdna
  z = reshape(vz(itypdna, :, :), lswathow, lbeamow);
  % wybranie fragmentu obrazu
  z1 = z(swathsektor, beamsektor);
  [fragm_sizx fragm_sizy] = size(z1);
  % detrending
  if detrending
    for iswath = 1 : fragm_sizx
      a = polyfit(1 : fragm_sizy, z1(iswath, :), 1);
      z1(iswath, :) = z1(iswath, :) - a(1) * (1:fragm_sizy) - a(2);
    end
  end  
  figure
  colormap([zeros(256,1)  (1/256:1/256:1)' ones(256,1)]);  % generacja palety kolorów
  mesh(beamsektor, swathsektor, z1)  % wyœwietlenie powierzchni dna w 3D
  set(gca, 'ZLim', z_zakres)  % ustawienie zakresu osi z (mo¿na z tym poeksperymentowaæ)
end  % itypdna = 1 : ltypowdna
