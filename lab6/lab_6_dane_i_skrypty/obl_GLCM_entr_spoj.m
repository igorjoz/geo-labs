% Obliczenie entropii i spójnoœci lokalnej dla prostok¹tnego fragmentu obrazu sonarowego z u¿yciem techniki GLCM
% Korzysta z parGLCM.M i GLCM.m.
N_GL = 8;  % liczba poziomów szaroœci
d = 1;  % przesuniêcie w obrazie (w pikselach) przy liczeniu macierzy wspólnych wyst¹pieñ
pierwsza_dana = 1;  % Od którego sondowania (wiersza w obrazie) zacz¹æ?
liczba_danych = 500;  % Dla ilu kolejnych sondowañ (wierszy w obrazie) dokonaæ obliczeñ?
kolumny = 91 : 100;  % Które kolumny pikseli ma obejmowaæ fragment obrazu?
liczba_wierszy = 30;  % Które wiersze ma obejmowaæ fragment obrazu?
load dane_bss
%   £aduje siê zmienna vbss - tablica 3-wymiarowa o rozmiarach: liczba
%     typów dna (= 4), liczba sondowañ (danych) (= 600), liczba wi¹zek
%     (wartoœci bss) w 1 sondowaniu (= 160).
[ltypowdna lswathow lbeamow] = size(vbss);
% Czyszczenie i przesuwanie (jeœli któraœ wartoœæ BSS = 0, to ustaw j¹ na minimaln¹ wartoœæ z
%   ca³ego zapisu, a nastêpnie dodaj tê minimaln¹ wartoœæ do wszystkich
%   danych).
calosciowemin = min(min(min(vbss)));
for itypdna = 1 : ltypowdna
  for idana = pierwsza_dana : pierwsza_dana - 1 + liczba_danych + liczba_wierszy  % Bierzemy pod uwagê wszystkie linie obrazu których "u¿yjemy".
    for ikolumna = 1 : lbeamow
      if vbss(itypdna, idana, ikolumna) == 0
        vbss(itypdna, idana, ikolumna) = calosciowemin;
      end 
    end  
  end
end
calosciowemaks = max(max(max(vbss)));
vbss = vbss - calosciowemin;  % obliczenie wartoœci entropii i spójnoœci lokalnej za pomoc¹ GLCM dla
%   fragmentów obrazu odpowiadaj¹cych poszczególnym swath'om
clear var_GLCM_entr var_GCLM_spoj
for itypdna = 1 : ltypowdna
  for idana = pierwsza_dana : pierwsza_dana - 1 + liczba_danych  % Zak³adamy, ¿e nie przekroczymy tu lswathow, z zapasem na liczbê wierszy.
    fragm_obrazu_do_stat_ = vbss(itypdna, idana : idana - 1 + liczba_wierszy, kolumny);  % Wybieramy odpowiedni fragment odpowiedniego obrazu.
    fragm_obrazu_do_stat = reshape(fragm_obrazu_do_stat_, liczba_wierszy, length(kolumny));  % Przekszta³camy na zwyk³¹ tablicê 2D.
    [entropia, spoj_lokal] = parGLCM(fragm_obrazu_do_stat, d, N_GL, 0, calosciowemaks - calosciowemin);
    var_GLCM_entr(itypdna, idana) = entropia;
    var_GLCM_spoj(itypdna, idana) = spoj_lokal;      
  end  % for idana
end  % for itypdna
