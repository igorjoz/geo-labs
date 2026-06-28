
load dane_bss;
[lplikow lswathow lbeamow] = size(vbss);
% wyœwietlenie obrazów
for iplik = 1 : lplikow
  bss = reshape(vbss(iplik, :, :), lswathow, lbeamow);
  figure
  imagesc(bss)
end  % iplik = 1 : lplik
