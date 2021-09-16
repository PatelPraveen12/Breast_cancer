<?php
$array = array("firstname" => "", "name" => "", "email" => "", "phone" => "", "message" => "", "firstnameError" => "", "nameError" => "", "phoneError" => "", "messageError" => "","isSuccess" => false);

$emailto = "test@test.com";

if ($_SERVER['REQUEST_METHOD'] == "POST") {

    $array['firstname'] = verifyInput($_POST['firstname']);
    $array['name'] = verifyInput($_POST['name']);
    $array['email'] = verifyInput($_POST['email']);
    $array['phone'] = verifyInput($_POST['phone']);
    $array['message'] = verifyInput($_POST['message']);
    $array['isSuccess'] = true;
    $emailText = "";
    

    if (empty($array['firstname'])) {
        $array['firstnameError'] = "je veux connaître ton prénom";
        $array['isSuccess'] = false;
    }else{
        $emailText .= "firstname: {$array['firstname']}\n";
    }

    if (empty($array['name'])) {
        $array['nameError'] = "Et même ton nom";
        $array['isSuccess'] = false;
    } else {
        $emailText .= "name: {$array['name']}\n";
    }

    if(!isEmail($array['email'])){
        $array['emailError'] = " Ce n'est pas un email valide !";
        $array['isSuccess'] = false;
    } else {
        $emailText .= "email: {$array['email']}\n";
    }

    if(!isPhone($array['phone'])){
        $array['phoneError'] = " Que des chiffres et des espaces ";
        $array['isSuccess'] = false;
    } else {
        $emailText .= "phone: {$array['phone']}\n";
    }

    if (empty($array['message'])) {
        $array['messageError'] = "Que veux tu me dire ?";
        $array['isSuccess'] = false;
    } else {
        $emailText .= "message: {$array['message']}\n";
    }

    if($array['isSuccess']){
        $headers = "From: {$array['firstname']} {$array['name']} <{$array['email']}>\r\n\Reply-To: {$array['email']}";
        mail($emailto, "un message de votre site",$emailText,$headers);
    }

    echo json_encode($array);

}

function verifyInput($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);

    return $data;
}

function isEmail($var){
    return filter_var($var, FILTER_VALIDATE_EMAIL);
}

function isPhone ($var){

    // expression régulière
    return preg_match("/^[0-9 ]*$/", $var);
}