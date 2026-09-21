# Полный набор локальных проверок в контейнере: ruff, mypy, pytest.
# Останавливается на первой ошибке — иначе непонятно, какая из трёх упала.
#
# Ключ -Force снимает отказ, когда на машине уже идут проверки (см. ниже).
# Ключ -NoBuild пропускает сборку образа. Заведён по случаю 06.09: Docker Hub не отдал
# фронтенд docker/dockerfile:1, сборка упала первой же строкой, и вместе с ней стали
# недоступны три проверки, которым сеть не нужна вовсе.
# Сборка остаётся обязательной по умолчанию, а не «предупреждением»: зелёный check.ps1
# единственный способ проверить работу целиком, и он обязан означать,
# что проверки прошли на образе из текущего uv.lock. Пропуск ключом — осознанное
# действие пользователя, и он виден первой строкой вывода.
#
# Имена переменных латиницей намеренно: Windows PowerShell 5.1 читает .ps1 в кодировке
# системы, и кириллический идентификатор превращается в синтаксическую ошибку.
# ErrorActionPreference намеренно не "Stop": docker пишет прогресс сборки в stderr,
# и при "Stop" PowerShell считает это ошибкой ещё до того, как команда вернёт код.
# Успех определяется по $LASTEXITCODE, а не по тому, что попало в stderr.
[CmdletBinding()]
param([switch]$NoBuild, [switch]$Force)

$ErrorActionPreference = "Continue"

# Вывод скрипта — на русском, а консоль Windows по умолчанию печатает его в кодировке
# OEM (866): сообщения о провале превращаются в «����� �� ��設�» ровно в тот момент,
# когда их надо прочитать. Кодировка задаётся здесь, а не в профиле пользователя: скрипт
# должен читаться на любой машине, включая чужую и CI.
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

# Отсутствие docker в PATH — не гипотетический случай: установщик Docker Desktop правит
# системный PATH, но уже открытые терминалы его не перечитывают. PowerShell при этом
# бросает CommandNotFoundException, НЕ трогая $LASTEXITCODE, поэтому проверка кода возврата
# такой провал пропускает и скрипт рапортует успех, не выполнив ни одной проверки.
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ПРОВАЛ: docker не найден в PATH." -ForegroundColor Red
    Write-Host "Если Docker Desktop только что установлен, перезапустите терминал." -ForegroundColor Yellow
    exit 1
}

# Docker Desktop на Windows часто оказывается в состоянии Stopped. Без этой проверки
# пользователь увидит невнятную ошибку про npipe вместо понятной причины.
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ПРОВАЛ: демон Docker не отвечает." -ForegroundColor Red
    Write-Host "Запустите Docker Desktop и дождитесь, пока он перейдёт в состояние Running." -ForegroundColor Yellow
    exit 1
}

# Проверки, уже идущие в соседней сессии, второй раз не запускаются. Участник держит
# две сессии на одном каталоге, и два pytest на одной машине не просто
# ждут друг друга: они делят процессор, и прогон растягивается втрое. 18.09 полный набор
# занял 13 минут при параллельной сессии и 4 мин 46 с в одиночку — разница не в тестах,
# а в соседе. Определяется по контейнерам проекта: `docker compose run` называет их
# `kinase-ifp-potential-<сервис>-run-<id>`.
$busy = @(docker ps --filter "name=kinase-ifp-potential-" --format "{{.Names}} ({{.Status}})")
if ($busy.Count -gt 0 -and -not $Force) {
    Write-Host "ОТКАЗ: на машине уже идут проверки или расчёты в контейнере." -ForegroundColor Red
    foreach ($c in $busy) { Write-Host "  $c" -ForegroundColor Yellow }
    Write-Host "Дождитесь их окончания: два прогона на одной машине делят процессор" -ForegroundColor Yellow
    Write-Host "и оба идут втрое дольше. Запустить всё равно: scripts/check.ps1 -Force" -ForegroundColor Yellow
    exit 1
}

# Общий секундомер: сумма по шагам отвечает на вопрос «сколько это вообще длится»,
# который до 16.09 не имел ответа нигде, кроме записок.
$total = [Diagnostics.Stopwatch]::StartNew()

function Invoke-Docker {
    param([string]$Label, [string[]]$Arguments, [string]$Hint)
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    if ($Hint) { Write-Host $Hint -ForegroundColor DarkGray }
    # Время печатается ПОСЛЕ шага, а не по ходу: вывод docker идёт прямо в консоль,
    # и заворачивать его в конвейер нельзя — в PowerShell 5.1 stderr внешней команды
    # в конвейере становится ErrorRecord, а stderr здесь не ошибка (см. шапку файла).
    $watch = [Diagnostics.Stopwatch]::StartNew()
    # Сброс намеренный: если внешняя команда не запустится вовсе, $LASTEXITCODE сохранит
    # значение от предыдущего вызова, и провал будет принят за успех.
    $global:LASTEXITCODE = $null
    & docker @Arguments
    $watch.Stop()
    $spent = "{0:mm\:ss}" -f $watch.Elapsed
    if ($null -eq $LASTEXITCODE) {
        Write-Host "ПРОВАЛ: $Label — команда docker не запустилась ($spent)." -ForegroundColor Red
        exit 1
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ПРОВАЛ: $Label за $spent (код $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "=== $Label — готово за $spent ===" -ForegroundColor Green
}

if ($NoBuild) {
    Write-Host "=== сборка образа: пропущена (-NoBuild) ===" -ForegroundColor Yellow
    Write-Host "Проверки идут на уже собранном образе. Менялись uv.lock или Dockerfile — соберите: docker compose build dev" -ForegroundColor Yellow
} else {
    Invoke-Docker "сборка образа" @("compose", "build", "dev")
}
Invoke-Docker "ruff"   @("compose", "run", "--rm", "dev", "ruff", "check", ".")
Invoke-Docker "mypy"   @("compose", "run", "--rm", "dev", "mypy", "src")
Invoke-Docker "pytest" @("compose", "run", "--rm", "dev", "pytest", "-q") `
    -Hint "Больше тысячи тестов, обычно 3-9 минут (измерено 203-523 с). До конца прогона вывода не будет — это нормально, прерывать не нужно."

$total.Stop()
Write-Host ("Все проверки пройдены за {0:mm\:ss}." -f $total.Elapsed) -ForegroundColor Green
