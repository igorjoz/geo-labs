% Rysowanie 2D plot'w wartoœci obliczonych parametrów
%zm1 = var_srednia;  % pierwszy parametr (--> oœ x)
%zm2 = var_spadek_dB;  % drugi parametr (--> oœ y)
%zm1 = var_GLCM_entr;  % pierwszy parametr (--> oœ x)
%zm2 = var_GLCM_spoj;  % drugi parametr (--> oœ y)
zm1 = var_stdh;
zm2 = var_sixph;
% Rozmiar zm1 i zm2 musi byæ taki sam.
tablicaKolorow = {'bx', 'go', 'y+', 'r*'};
[ltypowdna, lobiektow] = size(zm1);  

figure
hold on
for itypdna = 1 : ltypowdna
  zmx = zm1(itypdna, :);
  zmy = zm2(itypdna, :);
  plot(zmx, zmy, char(tablicaKolorow(itypdna)))  % Rysujemy rozk³¹d wartoœci dla kolejnego typu dna.
end  % itypdna = 1 : ltypdna
hold off
