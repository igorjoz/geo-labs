function macGLCM = GLCM(macierz, di, dj, N_GL, totalmin, totalmax)
% Oblicza macierz GLCM dla obrazu zawartego w macierz dla przesuniêcia (di, dj).
% d - przesuniêcie w obrazie (w pikselach) przy liczeniu macierzy wspólnych wyst¹pieñ
% N_GL - liczba poziomów szaroœci
% totalmin, totalmax - zak³¹dana minimalna i maksymalna wartoœæ piksela w
%   obrazie
if totalmin > totalmax
  disp('macGLCM: totalmin > totalmax, nie mo¿na liczyæ GLCM.')
  pause
end
% dj musi byæ >= 0.
[limac ljmac] = size(macierz);
if di > limac || dj > ljmac
  dizp(['macGLCM: Za du¿y offset. limac = ' limac ', ljmac = ' ljmac ', di = ' di ', dj = ' dj])
  pause
end
% kwantyzacja poziomów szaroœci
if (totalmin < totalmax)
  mac_kwant = floor(N_GL * (macierz - totalmin) / (totalmax - totalmin));
  mac_kwant = mac_kwant - (mac_kwant == N_GL);  % Dla N poziomów szaroœci: 0, 1, 2, ..., N-1 to co 
%  "trafi³o" do N-tego poziomu (bo by³o równe totalmax) powinno "zasiliæ"
%  N-1-ty poziom.
else
  mac_kwant = zeros(limac, ljmac);  
end
% obliczenie GLCM (w wersji symetrycznej)
macGLCM = zeros(N_GL);
for i = 1 - di*(di<0) : limac - di*(di>0)
  for j = 1 : ljmac - dj
    GL1 = mac_kwant(i, j);
    GL2 = mac_kwant(i + di, j + dj);
    i_GLCM = min(GL1, GL2) + 1;
    j_GLCM = max(GL1, GL2) + 1;
    macGLCM(i_GLCM, j_GLCM) = macGLCM(i_GLCM, j_GLCM) + 1;
  end  % for j    
end  % for i
macGLCM = macGLCM + macGLCM';
macGLCM = macGLCM / sum(sum(macGLCM));

