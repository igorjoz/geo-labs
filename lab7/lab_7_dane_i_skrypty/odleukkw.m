function odl_kw = odleuk(x1, x2)
% Oblicza kwadrat odleg³oœci euklidesowej pomiêdzy wektorami x1 i x2 w
%   przestrzeni length(x1)-wymiarowej (d³ugoœci obu wektorów musz¹ byæ
%   takie same).
odl_kw = sum((x1 - x2) .* (x1 - x2));
