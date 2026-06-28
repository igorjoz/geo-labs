function [entropia, spoj_lokal] = parGLCM(macierz, d, N_GL, totalmin, totalmax)
% Oblicza entropiê i spójnoœæ lokaln¹ dla uœrednionych wartoœci GLCM dla
% obrazu zawartego w macierz dla przesuniêæ: (d,0), (0,d), (d,d), (-d,d).
% d - przesuniêcie w obrazie (w pikselach) przy liczeniu macierzy wspólnych wyst¹pieñ
% N_GL - liczba poziomów szaroœci
% totalmin, totalmax - zak³¹dana minimalna i maksymalna wartoœæ piksela w
%   obrazie
macGLCM10 = GLCM(macierz, d, 0, N_GL, totalmin, totalmax);
macGLCM01 = GLCM(macierz, 0, d, N_GL, totalmin, totalmax);
macGLCM11 = GLCM(macierz, d, d, N_GL, totalmin, totalmax);
macGLCM_11 = GLCM(macierz, -d, d, N_GL, totalmin, totalmax);
macGLCM = (macGLCM10 + macGLCM01 + macGLCM11 + macGLCM_11) / 4;
entropia = 0;
spoj_lokal = 0;
for i = 1 : N_GL
  for j = 1 : N_GL
    if macGLCM(i, j) > 0
      entropia = entropia - macGLCM(i, j) * log10(macGLCM(i, j));
      spoj_lokal = spoj_lokal + macGLCM(i, j) / (1 + (i-j)*(i-j));
    end
  end  
end
