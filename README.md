# Projekt_machine_learning
#Dzmitry Barbash i Lizaveta Kameneva

!Opis problemu biznesowego!

Do prawidłowego funkcjonowania firmy potrzebują sprawnych urządzeń. Działanie niektórych urządzeń ma krytyczne znaczenie. Na przykład samolot nie wystartuje jeżeli jego silnik nie będzie sprawny w 100% procentach sprawny. Większość firm naprawia urządzenia w momencie, gdy przestają one już działać (nosi to nazwę strategii konserwacji korygującej). Naraża ona firmy na kosztowne przestoje i wysokie koszty niespodziewanych napraw. Dlatego niektóre firmy rozwijają strategię konserwacji proaktywnej, która polega na przewidywaniu, kiedy urządzenie ulegnie awarii, i wymianie tylko tych podzespołów, których poprawne działanie nie jest już pewne. 

Przyjrzyjmy się linii lotniczej Oceanic Arilines. Jednym z powodów opóźniania lotów są problemy przewoźnika z serwisowaniem silników samolotów. Ponieważ w firmie obowiązuje zasada zapobiegania awariom silników za wszelką cenę, ich stan techniczny jest regularnie sprawdzany. Jednak zużycie elementów silnika samolotowego zależy od wielu czynników, nie tylko od czasu jego pracy i odległości przelotów. W efekcie niektóre czynności serwisowe są przeprowadzane niepotrzebnie, a zdarza się, że przeciągający się przegląd powoduje opóźnienie odlotu.

Wszystkie samoloty linii są wyposażone w czujniki monitujące pracę urządzeń pokładowych, w tym silników. Firma ma przykładowe dane zarówno bezbłędnie działających silników, jak i silników, które uległy awarii. Naszym zadaniem jest zbudowanie modelu, który przewidzi czas pozostałej bezpiecznej eksploatacji RUL silników. Zbudowany model zostanie użyty do wdrożenia strategii proaktywnej konserwacji.

!Opis danych!

Dane zapisane są w dwóch plikach (załączone). Plik treningowy zawiera opis działania 100 silników tego samego typu. Praca każdego silnika jest przedstawiona w postaci 24 szeregów czasowych. Kluczem tych serii jest numer cyklu, a wartościami bieżące parametry pracy silnika i wyniki pomiarów jego działania. Chociaż dla wszystkich 100 silników pierwszym numerem jest liczba 1, ich pomiary zaczęły się w różnym czasie. Czyli pierwszy pomiar został przeprowadzony dla silników o różnym, nieznanym, nam stopniu zużycia. Wiemy natomiast, że na początku każdego cyklu wszystkie silniki działały prawidłowo oraz to, że z czasem ich działanie ulegało pogorszeniu. Po przekroczeniu określonej wartości granicznej parametrów pracy silnik  został uznany za wymagający przeglądu. Ostatni odnotowany pomiar należy uznać za moment, w którym silnik uległ awarii. Dla każdego silnika moment jego awarii może być innych, czyli długość opisujących różne silniki szeregów czasowych jest różna. Plik treningowy liczy ponad 20 tysięcy wierszy i prawie 30 kolumn. 
